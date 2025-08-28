# pylint: disable=arguments-differ
# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""VLLM module"""

import inspect
import os
from typing import Optional, Dict, List, TYPE_CHECKING
import copy

import torch
from transformers import AutoTokenizer, AutoConfig

try:
    from vllm import SamplingParams
    from vllm.config import LoadFormat
    from vllm.entrypoints.llm import LLM
    from vllm.distributed import parallel_state
    from vllm.executor.ray_utils import RayWorkerWrapper
    from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion
    from chatlearn.utils.global_vars import set_vllm_actors
    from chatlearn.utils.vllm_utils import initialize_vllm
    from chatlearn.utils.mappings.vllm_helpers import ShardedTensorInfo, build_sharded_info_for_vllm_model
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False

from chatlearn.utils.utils import get_full_proc_memory_info
from chatlearn.runtime.decorator import timeit
from .torch_module import TorchModule

if TYPE_CHECKING:
    from chatlearn.synchronizer.structs import BucketInfo


if HAVE_VLLM:
    # pylint: disable=unexpected-keyword-arg
    class VLLMModule(TorchModule, RayWorkerWrapper):
        """VLLMModule"""

        def __init__(self, name: str, args=None, replica_id: int=0, **kwargs):
            """The chatlearn wrapper for a vLLM model.

            Args:
                name (str): The name of this module
                args (Any, optional): The arguments passed to ChatLearn. Defaults to None.
                replica_id (int, optional): The replica id of this module. Defaults to 0.
                kwargs (Dict[str, Any]): The keyword arguments passed to `RayWorkerWrapper`. 
            """
            TorchModule.__init__(self, name, args=args, replica_id=replica_id)

            # avoid overwrite methods
            methods_class1 = {method[0] for method in inspect.getmembers(TorchModule, predicate=inspect.isfunction)}
            methods_class2 = {method[0] for method in inspect.getmembers(RayWorkerWrapper, predicate=inspect.isfunction)}
            common_methods = methods_class1.intersection(methods_class2)
            # common method is '__init__'
            assert common_methods == {'__init__'}, \
                f"Expected only '__init__' as common method for TorchModule and RayWorkerWrapper, but got {common_methods}"
            self.local_rank = 0

            assert CURRENT_VLLM_VERSION == VLLMVersion.v_0_8_5, "only vllm0.8.5 support, if you want to use previous vllm version, \
                    please git checkout 4ad5912306df5d4a814dc2dd5567fcb26f5d473b"
            if 'vllm_actor_type' in kwargs and 'worker' == kwargs['vllm_actor_type']:
                vllm_config = self.init_engine_args()
                RayWorkerWrapper.__init__(self, vllm_config=vllm_config, rpc_rank=kwargs['rpc_rank']) # pylint: disable=non-parent-init-called

            os.environ['VLLM_HOST_IP'] = self.get_address()

            self.tokenizer = None
            self._model = None
            self.llm = None
            self.model_config =  AutoConfig.from_pretrained(self.module_args.load)
            if self.model_config.architectures[0] == "Qwen3MoeForCausalLM":
                self.param_update_fn = self.update_weights_from_ipc_handles_qwen3_moe
            else:
                self.param_update_fn = self.update_weights_from_ipc_handles_naive

            self.set_vllm_pp_layer_partition()
            self._metric_prefix = 'rollout'

        def init_engine_args(self):
            dtype = self.module_args.get("dtype", "bfloat16")
            if self.module_args.get("fp16", False):
                dtype = "float16"

            load_format = self.module_args.get("load_format", LoadFormat.DUMMY)

            if self.module_args.get("apply_replica_id_to_seed", True):
                seed = self.module_args.get("seed", 0) + self.replica_id
            else:
                seed = self.module_args.get("seed", 0)

            from vllm.engine.arg_utils import AsyncEngineArgs # pylint: disable=import-outside-toplevel
            from vllm.usage.usage_lib import UsageContext # pylint: disable=import-outside-toplevel
            self.engine_args = AsyncEngineArgs(
                model=self.module_args['load'],
                tokenizer=self.module_args['load'],
                max_seq_len_to_capture=self.module_args.get("max_seq_len_to_capture", 32768),
                seed=seed,
                # load model: 'dummy' for megatron ckpt or mock weight; others for hf ckpt.
                load_format=load_format,
                # parallelism strategy
                tensor_parallel_size=self.module_args.tensor_model_parallel_size,
                pipeline_parallel_size=self.module_args.pipeline_model_parallel_size,
                dtype=dtype,
                # scheduling strategy
                max_num_seqs=self.module_args.generation_batch_size,
                max_num_batched_tokens = self.module_args.get("max_num_batched_tokens", None),
                num_scheduler_steps=self.module_args.get("num_scheduler_steps", 1),
                gpu_memory_utilization=self.module_args.get("gpu_memory_utilization", 0.90),
                # logger
                disable_log_requests=self.module_args.get("disable_log_requests", True),
                disable_log_stats=self.module_args.get("disable_log_stats", True),
                trust_remote_code=True,
                enforce_eager=self.module_args.get("enforce_eager", True),
                disable_custom_all_reduce=True,
                distributed_executor_backend="ray",
                enable_sleep_mode=True,
                swap_space=self.module_args.get("swap_space", 16))
            return self.engine_args.create_engine_config(usage_context=UsageContext.ENGINE_CONTEXT)

        def load_model(self):
            """Initialize `self._model` here. This function is implicitly called in
            `RayDistributedExecutor._init_workers_ray`
            """
            assert self.worker is not None, \
                    "please set env variables `VLLM_USE_RAY_SPMD_WORKER=1` and `VLLM_USE_RAY_COMPILED_DAG=1` first."
            self.worker.load_model()
            self._model = self.worker.model_runner.model

        def init(self):
            """
            set global variables, initialize distributed env.
            used in BaseEngine.setup
            """
            # set global var _ENABLE_CUSTOM_ALL_REDUCE to False
            parallel_state.set_custom_all_reduce(False)
            initialize_vllm(self.module_args)

        def setup(self):
            """Set up tokenizer.
            used in BaseEngine.setup and BaseModule.model_setup
            """
            super().setup()
            tokenizer = AutoTokenizer.from_pretrained(self.module_args['load'], trust_remote_code=True)
            tokenizer.tokenizer = tokenizer
            self.tokenizer = tokenizer

        @timeit()
        def setup_engine(self, workers):
            """setup vllm engine
            used in Environment.setup()
            """

            if self.llm is not None: # for evaluator not setup twice
                return
            # setup vllm engine in rank 0
            os.environ['VLLM_HOST_IP'] = self.get_address()
            set_vllm_actors(workers)

            dtype = self.module_args.get("dtype", "bfloat16")
            if self.module_args.get("fp16", False):
                dtype = "float16"

            load_format = self.module_args.get("load_format", LoadFormat.DUMMY)

            if self.module_args.get("apply_replica_id_to_seed", True):
                seed = self.module_args.get("seed", 0) + self.replica_id
            else:
                seed = self.module_args.get("seed", 0)
            self.llm = LLM(
                model=self.module_args.load,
                tokenizer=self.module_args.load,
                max_seq_len_to_capture=self.module_args.get("max_seq_len_to_capture", 32768),
                seed=seed,
                # load model: 'dummy' for megatron ckpt or mock weight; others for hf ckpt.
                load_format=load_format,
                # parallelism strategy
                tensor_parallel_size=self.module_args.tensor_model_parallel_size,
                pipeline_parallel_size=self.module_args.pipeline_model_parallel_size,
                dtype=dtype,
                # scheduling strategy
                max_num_seqs=self.module_args.generation_batch_size,
                max_num_batched_tokens = self.module_args.get("max_num_batched_tokens", None),
                num_scheduler_steps=self.module_args.get("num_scheduler_steps", 1),
                gpu_memory_utilization=self.module_args.get("gpu_memory_utilization", 0.90),
                # logger
                disable_log_requests=self.module_args.get("disable_log_requests", True),
                disable_log_stats=self.module_args.get("disable_log_stats", True),
                trust_remote_code=True,
                enforce_eager=self.module_args.get("enforce_eager", False),
                disable_custom_all_reduce=True,
                distributed_executor_backend="ray",
                enable_sleep_mode=True,
                swap_space=self.module_args.get("swap_space", 16))
            self.offload()

        def dump_parameters(self, dump_path_root):
            self.onload()
            self.llm.llm_engine.model_executor._run_workers("worker_dump_parameters", dump_path_root=dump_path_root)
            self.offload()

        def worker_dump_parameters(self, dump_path_root):
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            model = self.model
            if isinstance(model, list):
                model = model[0]

            dir_path = os.path.join(dump_path_root, str(tp_rank))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            self._logger.info(f"dump parameters to {dir_path}")
            for name, weight in self.model.state_dict().items():
                pt_file = os.path.join(dir_path, name)
                torch.save(weight.data.clone(), pt_file)

        def set_vllm_pp_layer_partition(self):
            pipeline_world_size = self.module_args.pipeline_model_parallel_size
            num_layers = self.model_config.num_hidden_layers
            remainder = num_layers % pipeline_world_size

            if not self.module_args.get("allow_padding_num_layers", None):
                assert remainder == 0, \
                    f"expect num_layers % pipeline_model_size == 0 when VLLM_PP_LAYER_PARTITION is not set. \
                    while num_layers = {num_layers} pipeline_model_size = {pipeline_world_size}"
                return

            if remainder > 0:
                assert not self.module_args.get("standalone_embedding_stage", False), \
                    "not support standalone embedding stage if allow_padding_num_layers is true"
                # pad num_layers to make num_layers % pipeline_model_parallel_size == 0
                num_layers_with_padding = num_layers - remainder + pipeline_world_size
            else:
                num_layers_with_padding = num_layers
            num_layers_without_padding = num_layers
            num_layers = num_layers_with_padding
            num_layers_per_stage_with_padding = (
                num_layers // pipeline_world_size)

            # Each stage gets a contiguous set of layers.
            if self.module_args.get("pipeline_layers", None) is not None:
                rank_sizes = self.module_args.get("pipeline_layers", None)
                assert isinstance(rank_sizes, list) and all(isinstance(ele, int) for ele in rank_sizes), \
                    f"pipeline_layers expected to be list, and num layer of each stage to be integer, while {rank_sizes}."
            else:
                rank_sizes = [num_layers_per_stage_with_padding] * pipeline_world_size
                num_padding = num_layers - num_layers_without_padding
                if num_padding > 0:
                    assert num_padding == 2, \
                        "Support num_padding_lsyers == 2 when applies inbalanced pp. Please set `args.pipeline_layers` for VLLMModule."

                for _index in range(-1, num_padding - 1):
                    rank_sizes[_index] -= 1
            assert len(rank_sizes) == pipeline_world_size

            # set env variable VLLM_PP_LAYER_PARTITION
            vllm_pp_layer_partition = ",".join([str(ele) for ele in rank_sizes])
            if os.getenv("VLLM_PP_LAYER_PARTITION", None) is not None:
                env_vllm_pp_layer_partition = os.getenv("VLLM_PP_LAYER_PARTITION", None)
                if vllm_pp_layer_partition != env_vllm_pp_layer_partition:
                    self._logger.warning(
                        f"expect VLLM_PP_LAYER_PARTITION to be {vllm_pp_layer_partition}, while {env_vllm_pp_layer_partition}")
            os.environ["VLLM_PP_LAYER_PARTITION"] = vllm_pp_layer_partition
            self._logger.info(f"Set VLLM_PP_LAYER_PARTITION={vllm_pp_layer_partition}")

        def _get_sampling_params(self, is_eval):
            temperature = 0.0
            if not self.module_args.get("use_beam_search", False):
                temperature = self.module_args.get("eval_temperature", 1.0) if is_eval else self.module_args.get(
                    "temperature", 1.0)
            top_p = self.module_args.get("eval_top_p", 1.0) if is_eval else self.module_args.get("top_p", 1.0)
            top_k = self.module_args.get("eval_top_k", -1) if is_eval else self.module_args.get("top_k", -1)
            min_p = self.module_args.get("eval_min_p", 0.0) if is_eval else self.module_args.get("min_p", 0.0)
            presence_penalty = self.module_args.get("eval_presence_penalty", 0.0) if is_eval else self.module_args.get(
                "presence_penalty", 0.0)
            frequency_penalty = self.module_args.get("eval_frequency_penalty", 0.0) if is_eval else self.module_args.get(
                "frequency_penalty", 0.0)
            repetition_penalty = self.module_args.get("eval_repetition_penalty", 1.0) if is_eval else self.module_args.get(
                "repetition_penalty", 1.0)
            stop = self.module_args.get("stop_token_list", None)
            if stop is not None and isinstance(stop, str):
                stop = stop.split(";")
            sampling_params = SamplingParams(
                n=self.module_args.get("n", 1),
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                ignore_eos=self.module_args.get("ignore_eos", False),
                stop=stop,
                logprobs=self.module_args.get("logprobs", None),
                detokenize=self.module_args.get("detokenize", False),
                prompt_logprobs=self.module_args.get("prompt_logprobs", None),
                skip_special_tokens=self.module_args.get('skip_special_tokens', True)
            )
            # VLLMVersion.v_0_3_0, VLLMVersion.v_0_5_1
            if hasattr(sampling_params, 'use_beam_search'):
                sampling_params.use_beam_search = self.module_args.get("use_beam_search", False)
            return sampling_params

        def update_weights_from_ipc_handles_qwen3_moe(self, reduce_data):
            n_expert = self.model_config.num_experts
            for name, reduced in reduce_data.items():
                rebuild_func, rebuild_args = reduced
                reconstructed_tensor = rebuild_func(*rebuild_args)
                if "group_mlp" in name:
                    reconstructed_tensor = torch.chunk(reconstructed_tensor, n_expert, dim=0)
                    for i in range(n_expert):
                        local_name = name.replace('group_mlp', f"experts.{i}")
                        self.model.load_weights([(local_name, reconstructed_tensor[i])])
                else:
                    self.model.load_weights([(name, reconstructed_tensor)])

        def update_weights_from_ipc_handles_naive(self, reduce_data):
            for name, reduced in reduce_data.items():
                rebuild_func, rebuild_args = reduced
                reconstructed_tensor = rebuild_func(*rebuild_args)
                self.model.load_weights([(name, reconstructed_tensor)])

        def update_weights_from_ipc_handles(self, reduce_data):
            self.param_update_fn(reduce_data)

        def generate_vllm(self, query, is_eval, iteration=0, is_first_run=True):
            # resume from stage checkpoint.
            outputs = self.load_stage_outputs(is_eval, iteration)
            if outputs is not None:
                return outputs
            # using for multi-round generate
            if is_first_run:
                self.llm.wake_up()

            # preprocess query
            prompt_key = "prompt"
            input_ids_key = "input_ids"
            seq_len = self.module_args.seq_length

            prompts = [q[prompt_key] for q in query]
            prompts_token_ids = [q[input_ids_key] for q in query]
            sampling_param = self._get_sampling_params(is_eval)
            sampling_params = []

            for prompt, prompt_token_ids_item in zip(prompts, prompts_token_ids):
                max_tokens = seq_len - len(prompt_token_ids_item)
                assert max_tokens > 0, f"{prompt} is larger than {seq_len}"
                sampling_param_item = copy.deepcopy(sampling_param)
                sampling_param_item.max_tokens = max_tokens
                sampling_params.append(sampling_param_item)

            outputs = self.llm.generate(prompt_token_ids = prompts_token_ids,
                                        sampling_params = sampling_params,
                                        use_tqdm = True)

            # save stage outputs for resume.
            self.save_stage_outputs(is_eval, outputs, iteration)
            return outputs

        def peak_memory(self):
            """
            :meta private:
            """
            self._peak_memory = max(self._peak_memory, torch.cuda.max_memory_allocated() / (1024 ** 3))
            return self._peak_memory

        @property
        def data_parallel_size(self):
            """
            :meta private:
            """
            return 1

        @property
        def data_parallel_rank(self):
            """
            :meta private:
            """
            return 0

        @property
        def model(self):
            return self._model

        @property
        def is_engine(self):
            return self.llm

        @timeit()
        def offload(self): # is_param_sync=True
            """
            offload weights
            """
            if self.module_args.free_gpu_memory.offload_weights and self.is_engine:
                self._logger.info(f"llm_engine.sleep before: {get_full_proc_memory_info('before llm_engine.sleep')}")
                self.llm.sleep()
                self._logger.info(f"llm_engine.sleep after: {get_full_proc_memory_info('after llm_engine.sleep')}")

        @timeit()
        def onload(self, tags: Optional[list[str]] = None):
            """
            onload weights
            Wake up the engine from sleep mode. See the :meth:`sleep` method
            for more details.

            Args:
                tags: An optional list of tags to reallocate the engine memory
                    for specific memory allocations. Values must be in
                    ("weights", "kv_cache",). If None, all memory is reallocated.
                    wake_up should be called with all tags (or None) before the
                    engine is used again.
            """
            if self.module_args.free_gpu_memory.offload_weights and self.is_engine:
                self._logger.info(f"llm_engine.wake_up before: {get_full_proc_memory_info('before llm_engine.wake_up')}")
                self.llm.wake_up(tags)
                self._logger.info(f"llm_engine.wake_up after: {get_full_proc_memory_info('after llm_engine.wake_up')}")

        @torch.no_grad()
        def map_local_param_name_to_global(self):
            """Map names of weights on each rank to a unique name.
            
            Returns:
                List[str]: A list of unique global names for each weight 
            on this rank.
            """
            names = list(self.model.state_dict().keys())
            self.global_name_to_local_name = {n: n for n in names}
            return names

        @torch.no_grad()
        def get_parameter_metadata(self) -> Dict[str, ShardedTensorInfo]:
            """Collect parameter shape info of this rank
            """
            if self.local_name_to_param_id is None:
                raise ValueError("Call set_param_id before call this function")

            infos = {}
            for name, sharded_info in build_sharded_info_for_vllm_model(self.model).items():
                param_id = self.local_name_to_param_id[name]
                sharded_info.param_id = param_id
                infos[param_id] = sharded_info
            return infos

        @torch.no_grad()
        def parameter_sync(self):
            """Perform parameter synchronization on this worker."""
            if self.synchronizer is None:
                raise ValueError("Synchronizer is not initialized.")
            param_id_to_parameters = {}
            for name, weight in self.model.state_dict().items():
                if name not in self.local_name_to_param_id:
                    continue
                param_id_to_parameters[self.local_name_to_param_id[name]] = weight

            self.param_id_to_parameters = param_id_to_parameters
            self.synchronizer.parameter_sync()
            self.param_id_to_parameters = None

        def update_weights_from_buckets(self, buckets: List[Optional['BucketInfo']]):
            for bucket in buckets:
                if bucket is None:
                    continue
                offset = 0
                if bucket.buffer is None:
                    raise ValueError("Attempt to read from a bucket without buffer")
                for offset, sharded_tensor_info in bucket.recv_layout:
                    shard = sharded_tensor_info.index(self.param_id_to_parameters[sharded_tensor_info.param_id])
                    comm_dtype = sharded_tensor_info.dtype
                    numel = shard.numel()
                    byte_data = bucket.buffer[offset: offset + numel * comm_dtype.itemsize]
                    # NOTE: if shard.dtype != comm_dtype, an implicit datatype conversion will happen
                    shard.copy_(byte_data.view(comm_dtype).view(shard.shape))
                    offset += numel * comm_dtype.itemsize

else:
    class VLLMModule(TorchModule):
        """Dummy Module for vLLM. Note that it is not the subclass of ray worker"""
        # pylint: disable=super-init-not-called, abstract-method
        def __init__(self, name: str, args=None, replica_id: int=0, **kwargs):
            raise SystemError("vLLM is not installed")
