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
"""VLLM module v2"""

import gc
import inspect
import os

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import LoadFormat
from vllm.entrypoints.llm import LLM
from vllm.executor.ray_utils import RayWorkerWrapper

from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion
from chatlearn.utils.global_vars import set_vllm_actors
from chatlearn.utils.vllm_import_helper import parallel_state
from chatlearn.utils.vllm_import_helper import get_pipeline_model_parallel_rank
from chatlearn.utils.vllm_import_helper import TextTokensPrompt
from chatlearn.utils.vllm_utils import initialize_vllm
from .torch_module import TorchModule
try:
    from .megatron.memory_manager import InferenceMemoryManager
except ImportError:
    InferenceMemoryManager = None

class VLLMModuleV2(TorchModule, RayWorkerWrapper):
    """VLLMModuleV2"""

    def __init__(self, *args, **kwargs):
        TorchModule.__init__(self, *args)
        # avoid overwrite methods
        methods_class1 = {method[0] for method in inspect.getmembers(TorchModule, predicate=inspect.isfunction)}
        methods_class2 = {method[0] for method in inspect.getmembers(RayWorkerWrapper, predicate=inspect.isfunction)}
        common_methods = methods_class1.intersection(methods_class2)
        # common method is '__init__'
        assert common_methods == {'__init__'}, \
            f"Expected only '__init__' as common method for TorchModule and RayWorkerWrapper, but got {common_methods}"
        self.local_rank = 0
        if CURRENT_VLLM_VERSION == VLLMVersion.v_0_6_3:
            if 'worker_module_name' in kwargs and 'worker_class_name' in kwargs:
                RayWorkerWrapper.__init__(self, **kwargs) # pylint: disable=non-parent-init-called
        else:
            if 'vllm_actor_type' in kwargs and 'worker' == kwargs['vllm_actor_type']:
                vllm_config = self.init_engine_args()
                RayWorkerWrapper.__init__(self, vllm_config=vllm_config) # pylint: disable=non-parent-init-called

        os.environ['VLLM_HOST_IP'] = self.get_address()

        self.tokenizer = None
        self._model = None
        self.llm = None
        self.set_vllm_pp_layer_partition()

    def add_extra_args(self, parser):
        """
        Add extra arguments for vllm.

        Args
        ----
        parser : ArgumentParser
            Add extra arguments.
        """
        group = parser.add_argument_group(title='vLLM extra arguments')
        group.add_argument('--distributed-backend', default='nccl',
                           choices=['nccl', 'gloo'],
                           help='Which backend to use for distributed training.')
        group.add_argument('--distributed-timeout-minutes', type=int, default=10,
                           help='Timeout minutes for torch.distributed.')
        return parser

    def init_engine_args(self):
        dtype = self.model_args.get("dtype", "bfloat16")
        if self.model_args.get("fp16", False):
            dtype = "float16"

        load_format = self.model_args.get("vllm_load_format", LoadFormat.DUMMY)
        if load_format == LoadFormat.DUMMY:
            self.model_args["need_load_ckpt"] = self.src_parameter_model is None
            model_loader_extra_config = self.model_args
        else:
            model_loader_extra_config = None

        if self.model_args.get("apply_replica_id_to_seed", True):
            seed = self.model_args.get("seed", 0) + self.replica_id
        else:
            seed = self.model_args.get("seed", 0)

        from vllm.engine.arg_utils import AsyncEngineArgs # pylint: disable=import-outside-toplevel
        from vllm.usage.usage_lib import UsageContext # pylint: disable=import-outside-toplevel
        self.engine_args = AsyncEngineArgs(
            model=self.model_args['tokenizer'],
            tokenizer=self.model_args['tokenizer'],
            max_seq_len_to_capture=self.model_args.get("max_seq_len_to_capture", 32768),
            seed=seed,
            # load model: 'dummy' for megatron ckpt or mock weight; others for hf ckpt.
            load_format=load_format,
            model_loader_extra_config=model_loader_extra_config,
            # parallelism strategy
            tensor_parallel_size=self.module_args.tensor_model_parallel_size,
            pipeline_parallel_size=self.module_args.pipeline_model_parallel_size,
            dtype=dtype,
            # scheduling strategy
            max_num_seqs=self.module_args.generation_batch_size,
            max_num_batched_tokens = self.model_args.get("max_num_batched_tokens", None),
            num_scheduler_steps=self.model_args.get("num_scheduler_steps", 1),
            gpu_memory_utilization=self.model_args.get("gpu_memory_utilization", 0.90),
            # logger
            disable_log_requests=self.model_args.get("disable_log_requests", True),
            disable_log_stats=self.model_args.get("disable_log_stats", True),
            trust_remote_code=True,
            enforce_eager=self.model_args.get("enforce_eager", False),
            disable_custom_all_reduce=True,
            distributed_executor_backend="ray")
        return self.engine_args.create_engine_config(usage_context=UsageContext.ENGINE_CONTEXT)


    def init(self):
        """
        :meta private:
        """
        parallel_state.set_custom_all_reduce(False)
        initialize_vllm(extra_args_provider=self.add_extra_args,
                        ignore_unknown_args=True,
                        args_dict=self.model_args)

    def setup(self):
        """Set up tokenizer."""
        super().setup()
        tokenizer = AutoTokenizer.from_pretrained(self.model_args['tokenizer'])
        tokenizer.tokenizer = tokenizer
        self.tokenizer = tokenizer

    def setup_vllm(self, workers):
        if self.llm is not None: # for evaluator
            return
        # setup vllm engine in rank 0
        os.environ['VLLM_HOST_IP'] = self.get_address()
        set_vllm_actors(workers)

        dtype = self.model_args.get("dtype", "bfloat16")
        if self.model_args.get("fp16", False):
            dtype = "float16"

        load_format = self.model_args.get("vllm_load_format", LoadFormat.DUMMY)
        if load_format == LoadFormat.DUMMY:
            self.model_args["need_load_ckpt"] = self.src_parameter_model is None
            model_loader_extra_config = self.model_args
        else:
            model_loader_extra_config = None

        if self.model_args.get("apply_replica_id_to_seed", True):
            seed = self.model_args.get("seed", 0) + self.replica_id
        else:
            seed = self.model_args.get("seed", 0)

        self.llm = LLM(
            model=self.model_args['tokenizer'],
            tokenizer=self.model_args['tokenizer'],
            max_seq_len_to_capture=self.model_args.get("max_seq_len_to_capture", 32768),
            seed=seed,
            # load model: 'dummy' for megatron ckpt or mock weight; others for hf ckpt.
            load_format=load_format,
            model_loader_extra_config=model_loader_extra_config,
            # parallelism strategy
            tensor_parallel_size=self.module_args.tensor_model_parallel_size,
            pipeline_parallel_size=self.module_args.pipeline_model_parallel_size,
            dtype=dtype,
            # scheduling strategy
            max_num_seqs=self.module_args.generation_batch_size,
            max_num_batched_tokens = self.model_args.get("max_num_batched_tokens", None),
            num_scheduler_steps=self.model_args.get("num_scheduler_steps", 1),
            gpu_memory_utilization=self.model_args.get("gpu_memory_utilization", 0.90),
            # logger
            disable_log_requests=self.model_args.get("disable_log_requests", True),
            disable_log_stats=self.model_args.get("disable_log_stats", True),
            trust_remote_code=True,
            enforce_eager=self.model_args.get("enforce_eager", False),
            disable_custom_all_reduce=True,
            distributed_executor_backend="ray")
        self.llm.llm_engine.model_executor._run_workers("init_memory_manager")
        self.offload_for_workers()
        self.empty_cuda_graph_for_workers()
        self.empty_cache_for_workers()

    def init_memory_manager(self):
        if self.module_args.offload_weights:
            if InferenceMemoryManager is None:
                raise Exception("Import InferenceMemoryManager failed, you may need to set right Megatron path first.")
            self._memory_manager = InferenceMemoryManager(
                self.model,
                self.runtime_args.bucket_size_mb_in_memory_manager,
            )

    def set_vllm_pp_layer_partition(self):
        pipeline_world_size = self.module_args.pipeline_model_parallel_size
        num_layers = self.model_args.get("num_layers")
        remainder = num_layers % pipeline_world_size

        if not self.model_args.get("allow_padding_num_layers", None):
            assert remainder == 0, \
                f"expect num_layers % pipeline_model_size == 0 when VLLM_PP_LAYER_PARTITION is not set. \
                while num_layers = {num_layers} pipeline_model_size = {pipeline_world_size}"
            return

        if remainder > 0:
            assert not self.model_args.get("standalone_embedding_stage", False), \
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
        if self.model_args.get("pipeline_layers", None) is not None:
            rank_sizes = self.model_args.get("pipeline_layers", None)
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
        if not self.model_args.get("use_beam_search"):
            temperature = self.model_args.get("eval_temperature", 1.0) if is_eval else self.model_args.get(
                "temperature", 1.0)
        top_p = self.model_args.get("eval_top_p", 1.0) if is_eval else self.model_args.get("top_p", 1.0)
        top_k = self.model_args.get("eval_top_k", -1) if is_eval else self.model_args.get("top_k", -1)
        presence_penalty = self.model_args.get("eval_presence_penalty", 0.0) if is_eval else self.model_args.get(
            "presence_penalty", 0.0)
        frequency_penalty = self.model_args.get("eval_frequency_penalty", 0.0) if is_eval else self.model_args.get(
            "frequency_penalty", 0.0)
        repetition_penalty = self.model_args.get("eval_repetition_penalty", 1.0) if is_eval else self.model_args.get(
            "repetition_penalty", 1.0)
        stop = self.model_args.get("stop_token_list", None)
        if isinstance(stop, str):
            stop = stop.split(";")
        sampling_params = SamplingParams(
            n=self.model_args.get("n"),
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ignore_eos=self.model_args.get("ignore_eos"),
            stop=stop,
            logprobs=1,
            prompt_logprobs=self.model_args.get("prompt_logprobs", None),
            skip_special_tokens=False
        )
        # VLLMVersion.v_0_3_0, VLLMVersion.v_0_5_1
        if hasattr(sampling_params, 'use_beam_search'):
            sampling_params.use_beam_search = self.model_args.get("use_beam_search")
        return sampling_params

    def _convert_v1_inputs(self, prompts, prompt_token_ids):
        num_requests = len(prompts)
        assert num_requests == len(prompt_token_ids), \
            ("The lengths of prompts and prompt_token_ids must be the same.")

        inputs = []
        for i in range(num_requests):
            if prompts[i] is None:
                assert isinstance(prompt_token_ids[i], List[int]), \
                    f"Expect prompt_token_ids[{i}] is List[int] when prompt is None, while {prompt_token_ids[i]}."
            if prompt_token_ids[i] is None:
                assert isinstance(prompts[i], str), \
                    f"Expect prompts[{i}] is a string when prompt_token_ids is None, while {prompts[i]}."
            item = TextTokensPrompt(
                prompt=prompts[i],
                prompt_token_ids=prompt_token_ids[i])
            inputs.append(item)

        return inputs

    def preprocess_inputs(self, query, is_eval):
        prompt_key = self.model_args.get("vllm_prompt_key", "prompt")
        input_ids_key = self.model_args.get("vllm_input_ids_key", "input_ids")

        prompts = query[prompt_key]
        prompts_token_ids = query[input_ids_key]
        seq_len = self.model_args.get("seq_length")
        parsed_prompts = []
        sampling_params = []
        for i, prompt in enumerate(prompts):
            prompt_token_ids = prompts_token_ids[i]
            if 'sampling_param' in query:
                sampling_param = query['sampling_param'][i]
            else:
                sampling_param = self._get_sampling_params(is_eval)
                if not self.model_args.get("new_token_limit", False):
                    max_tokens = seq_len - len(prompt_token_ids)
                else:
                    max_tokens = self.model_args.get("max_new_tokens")
                    assert max_tokens < seq_len, "max_new_tokens must less than seq length."
                sampling_param.max_tokens = max_tokens
            item = self._convert_v1_inputs(
                prompts=[prompt],
                prompt_token_ids=[prompt_token_ids],
            )[0]
            parsed_prompts.append(item)
            sampling_params.append(sampling_param)

        return parsed_prompts, sampling_params

    def run_vllm(self, parsed_prompts, sampling_params):
        outputs = self.llm.generate(
            parsed_prompts,
            sampling_params,
            use_tqdm=True
        )
        return outputs

    def generate_vllm(self, query, is_eval, is_first_run=True):
        if is_first_run: # using for multi-round generate
            self.reinit_cache_engine()
        parsed_prompts, sampling_params = self.preprocess_inputs(query, is_eval)

        outputs = self.run_vllm(parsed_prompts, sampling_params)
        return outputs

    def is_last_rank(self):
        return True

    def num_layers(self):
        """
        :meta private:
        """
        return self.llm.llm_engine.model_config.hf_config.num_hidden_layers

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
        if self._model is None:
            assert self.worker is not None, \
                "please set env variables `VLLM_USE_RAY_SPMD_WORKER=1` and `VLLM_USE_RAY_COMPILED_DAG=1` first."
            self._model = self.worker.model_runner.model
        return self._model

    def tensor_parallel_rank(self):
        """
        :meta private:
        """
        return parallel_state.get_tensor_model_parallel_rank()

    def pipeline_parallel_rank(self):
        """
        :meta private:
        """
        return get_pipeline_model_parallel_rank()

    def tensor_model_parallel_size(self):
        return self.tensor_and_expert_model_parallel_size()

    def expert_model_parallel_size(self):
        return 1

    def tensor_and_expert_model_parallel_size(self):
        """
        get tensor_and_expert_model_parallel_size
        :meta private:
        """
        # vLLM not supported to enable expert parallel size
        # thus: tensor_and_expert_model_parallel_size = tensor_parallel_size
        return parallel_state.get_tensor_model_parallel_world_size()

    def model_setup_for_workers(self):
        self.llm.llm_engine.model_executor._run_workers("model_setup")

    # pylint: disable=unused-argument
    def offload_for_workers(self,
                            to_onload_weights=None,
                            to_build_grad_buffers=None,
                            to_onload_main_weights=None,
                            to_onload_optimizer_states=None):
        """
        call offload for all workers
        """
        self.llm.llm_engine.model_executor._run_workers("offload")

    def onload_for_workers(self,
                           to_onload_weights=None,
                           to_build_grad_buffers=None,
                           to_onload_main_weights=None,
                           to_onload_optimizer_states=None):
        """
        call onload for all workers
        """
        self.llm.llm_engine.model_executor._run_workers("onload")

    def empty_cache_for_workers(self):
        """
        call empty cache for all workers
        """
        self.llm.llm_engine.model_executor._run_workers("empty_cache")

    def empty_cuda_graph_for_workers(self):
        """
        call empty cuda_graph for all workers
        """
        self.llm.llm_engine.model_executor._run_workers("empty_cuda_graph")

    def offload_weights(self):
        """
        offload weights
        """
        if self.module_args.offload_weights:
            self._memory_manager.offload_weights()

    def onload_weights(self):
        """
        onload weights
        """
        if self.module_args.offload_weights:
            self._memory_manager.onload_weights()

    def empty_cache(self):
        if self.worker.gpu_cache is not None:
            for ele in self.worker.gpu_cache: # pylint: disable=unused-variable
                ele = None
            self.worker.gpu_cache = None # pylint: disable=access-member-before-definition

        if hasattr(self.worker, "cache_engine") and self.worker.cache_engine is not None:
            for c_e in self.worker.cache_engine:
                c_e.cpu_cache = None
                c_e.gpu_cache = None
            self.worker.cache_engine = None

        self.clear_cache()

    def clear_cache(self):
        if not self.timers("gc").started_:
            self.timers("gc").start()
        gc.collect()
        self.timers("gc").stop()

        super().empty_cache()

    def empty_cuda_graph(self):
        if self.worker.model_runner.graph_runners is not None:
            len_graph_runners = len(self.worker.model_runner.graph_runners)
            for graph_runner in self.worker.model_runner.graph_runners:
                for _, runner in graph_runner.items():
                    runner.input_buffers = {}
                    runner.output_buffers = {}
                    runner._graph = None
            for i in range(len_graph_runners):
                self.worker.model_runner.graph_runners[i] = {}
                self.worker.model_runner.graph_memory_pool = None

    def reinit_cache_engine(self):
        # reinit cache engine
        self.llm.llm_engine.model_executor._run_workers("clear_cache")
        self.llm.llm_engine._initialize_kv_caches()
        self.llm.llm_engine.model_executor._run_workers("clear_cache")
