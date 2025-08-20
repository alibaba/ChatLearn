# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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
"""SGLang Moudle"""
import warnings
import os
import math
from typing import Optional, List, TYPE_CHECKING, Tuple, Dict
import copy
import asyncio
import traceback
import multiprocessing as mp

import ray
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer

from chatlearn.utils.utils import get_full_proc_memory_info
from chatlearn.runtime.decorator import timeit
from .torch_module import TorchModule

try:
    import sglang
    from sglang.srt.managers.tokenizer_manager import (
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
        UpdateWeightsFromTensorReqInput,
    )
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.model_executor.model_runner import LocalSerializedTensor
    from sglang.srt.utils import set_prometheus_multiproc_dir, set_ulimit, assert_pkg_version, get_bool_env_var, is_cuda
    def _set_envs_and_config(server_args):
        # Set global environments
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["NCCL_CUMEM_ENABLE"] = str(int(server_args.enable_symm_mem))
        if not server_args.enable_symm_mem:
            os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
        os.environ["CUDA_MODULE_LOADING"] = "AUTO"

        # Set prometheus env vars
        if server_args.enable_metrics:
            set_prometheus_multiproc_dir()

        # Set ulimit
        set_ulimit()

        # Check flashinfer version
        if server_args.attention_backend == "flashinfer":
            assert_pkg_version(
                "flashinfer_python",
                "0.2.11.post3",
                "Please uninstall the old version and "
                "reinstall the latest version by following the instructions "
                "at https://docs.flashinfer.ai/installation.html.",
            )
        if is_cuda() and not get_bool_env_var("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"):
            assert_pkg_version(
                "sgl-kernel",
                "0.3.5",
                "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
            )

        # if True:  # Keep this check for internal code compatibility
        #     # Register the signal handler.
        #     # The child processes will send SIGQUIT to this process when any error happens
        #     # This process then clean up the whole process tree
        #     # Note: This sigquit handler is used in the launch phase, and may be replaced by
        #     # the running_phase_sigquit_handler in the tokenizer manager after the grpc server is launched.
        #     def launch_phase_sigquit_handler(signum, frame):
        #         logger.error(
        #             "Received sigquit from a child process. It usually means the child failed."
        #         )
        #         kill_process_tree(os.getpid())

        #     signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)

        # Set mp start method
        mp.set_start_method("spawn", force=True)
    sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config
except Exception:
    traceback.print_exc()
    warnings.warn("SGLang is not installed.")

if TYPE_CHECKING:
    from chatlearn.synchronizer.structs import BucketInfo

# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class AsyncEngine(Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # default to use dummy load format, which need to reload weights in first time
        # self._need_reload = True

    async def release_memory_occupation(self, tags: Optional[list[str]] = None):
        """Release GPU occupation temporarily."""
        if tags is None:
            obj = ReleaseMemoryOccupationReqInput()
        else:
            obj = ReleaseMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None):
        """Resume GPU occupation."""
        # because __init__ is a sync method, it can not call the async release_memory_occupation
        # have to move release_memory_occupation from __init__ to here
        # For multi-stage awake, we run release weight and kv_cache when we resume weights for the first time.
        # if self._need_reload:
        #     print("debughh check offload")
        #     await self.release_memory_occupation()
        #     self._need_reload = False

        if tags is None:
            obj = ResumeMemoryOccupationReqInput()
        else:
            obj = ResumeMemoryOccupationReqInput(tags=tags)

        return await self.tokenizer_manager.resume_memory_occupation(obj, None)

    # async def update_weights_from_tensor(self, serialized_named_tensors, flush_cache):
    #     from sglang.srt.managers.tokenizer_manager import UpdateWeightsFromTensorReqInput
    #     update_weights_request = UpdateWeightsFromTensorReqInput(serialized_named_tensors = serialized_named_tensors, flush_cache=flush_cache)
    #     return await self.tokenizer_manager.update_weights_from_tensor(update_weights_request, None)

    async def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be false
        to avoid duplicated cache cleaning operation."""
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(self.server_args.tp_size)
            ],
            load_format=load_format,
            flush_cache=flush_cache,
        )
        return await self.tokenizer_manager.update_weights_from_tensor(obj, None)

    async def flush_cache(self):
        return await self.tokenizer_manager.flush_cache()

    async def abort_request(self, rid: str = "", abort_all: bool = False):
        """Abort a specific request or all requests.

        Args:
            rid: The request ID to abort. If empty and abort_all is False, no action is taken.
            abort_all: If True, abort all running requests regardless of rid.
        """
        return self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)

class SGLangModule(TorchModule):
    """SGLangModule"""
    # pylint: disable=abstract-method

    def __init__(self, name: str, args=None, replica_id: int=0):
        """The chatlearn wrapper for a sglang model.
        """
        super().__init__(name, args=args, replica_id=replica_id)
        self.tensor_model_parallel_size = self.module_args.tensor_model_parallel_size
        # get gpu_per_node used for setup sglang
        resource = ray.nodes()[0]['Resources']
        self.gpu_per_node = int(resource['GPU'])
        self.llm = None
        self._metric_prefix = 'sglang_inference'


    def init(self):
        """
        initialize distributed env
        """
        # we need cpu process group for communicate ipc handle
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")

        # init cpu_mesh
        self.cpu_mesh = init_device_mesh(device_type="cpu",
                                        mesh_shape=(self.tensor_model_parallel_size,),
                                        mesh_dim_names=["tp"])
        self._tp_rank = self.cpu_mesh['tp'].get_local_rank()
        self._tp_size = self.cpu_mesh["tp"].size()
        torch.cuda.get_device_capability()

        visible_devices = [None] * self.cpu_mesh.size()

        torch.distributed.all_gather_object(
            visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], self.cpu_mesh.get_group("tp")
        )
        self.visible_devices_set = set(",".join(visible_devices).split(","))

        # used for init sglang engine in ray actor
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))
        dist.barrier(group=self.cpu_mesh.get_group())

    def setup(self):
        super().setup()
        # tokenizer = AutoTokenizer.from_pretrained(self.module_args['load'], trust_remote_code=True)
        # tokenizer.tokenizer = tokenizer
        # self.tokenizer = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.module_args['load'], trust_remote_code=True)
        

    @timeit("setup_sglang")
    def setup_sglang(self):
        if self.llm is not None: # for evaluator not setup twice
            return
        nnodes_per_replica = math.ceil(self.tensor_model_parallel_size / self.gpu_per_node)
        if nnodes_per_replica > 1:
            dist_init_addr = f"{os.environ['MASTER_ADDR']}:{os.environ['SGLANG_NCCL_PORT']}"
        else:
            dist_init_addr = None

        load_format = self.module_args.get("load_format", "dummy")
        tp_size_per_node = self._tp_size // nnodes_per_replica
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        dtype = self.module_args.get("dtype", "bfloat16")

        if first_rank_in_node:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self.llm = AsyncEngine(
                model_path=self.module_args['load'],
                dtype=dtype,
                mem_fraction_static=self.module_args.get("gpu_memory_utilization", 0.85),
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes_per_replica,
                trust_remote_code=True,
                port=40000 + self.replica_id,
                nccl_port=int(os.environ["SGLANG_NCCL_PORT"]),
                mm_attention_backend="fa3",
                attention_backend="fa3",
                skip_tokenizer_init=True,
                disable_cuda_graph=self.module_args.get("enforce_eager", False)
            )

        # this two flag used for avoid onload, offload twice
        self.kv_cache_onloaded = True
        self.weight_onloaded =True

        # because __init__ is a sync method, it can not call the async release_memory_occupation
        # have to move release_memory_occupation from __init__ to here
        # For multi-stage awake, we run release weight and kv_cache when we resume weights for the first time.
        self.need_offload = True
        # await self.offload_weights()


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

        sampling_params = {
            "n": 1,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "ignore_eos": self.module_args.get("ignore_eos", False),
            "stop": stop,
            # "logprobs": self.module_args.get("logprobs", 1), # not found
            # "detokenize": self.module_args.get("detokenize", False),
            # "prompt_logprobs": self.module_args.get("prompt_logprobs", None),
            "skip_special_tokens": self.module_args.get('skip_special_tokens', True)
        }

        return sampling_params

    def is_last_rank(self):
        return True

    def preprocess_data(self, query: List[Dict], is_eval: bool):
        """
        generate sampling parameter query-wise
        """
        seq_len = self.module_args.seq_length

        prompts = [q["prompt"] for q in query]
        prompts_token_ids = [q["input_ids"] for q in query]
        sampling_param = self._get_sampling_params(is_eval)
        sampling_params = []

        for prompt, prompt_token_ids_item in zip(prompts, prompts_token_ids):
            max_tokens = seq_len - len(prompt_token_ids_item)
            assert max_tokens > 0, f"{prompt} is larger than {seq_len}"
            sampling_param_item = copy.deepcopy(sampling_param)
            sampling_param_item['max_new_tokens'] = max_tokens
            sampling_params.append(sampling_param_item)
        return prompts_token_ids, sampling_params

    async def generate(self, query: List[Dict], is_eval: bool) -> List[Dict]:
        outputs = None
        if self.is_engine:
            prompts_token_ids, sampling_params = self.preprocess_data(query, is_eval)
            # outputs = self.llm.generate(input_ids=prompts_token_ids,
            #                             sampling_params=sampling_params)

            outputs = await self.llm.async_generate(
                    prompt=None,  # because we have already convert it to prompt token id
                    sampling_params=sampling_params,
                    return_logprob=True,
                    input_ids=prompts_token_ids
                )
        # self.flush_cache()
        return outputs

    async def update_weights_from_ipc_handles(self, reduce_data):

        # pylint: disable-next=import-outside-toplevel
        for index, (name, serialized_tensor) in enumerate(reduce_data.items()):
            if self.is_engine:
                gathered_serialized_tensors = [None] * self._tp_size
            else:
                gathered_serialized_tensors = None

            dist.gather_object(
                obj=serialized_tensor,
                object_gather_list=gathered_serialized_tensors,
                dst=self.cpu_mesh["tp"].mesh.tolist()[0],
                group=self.cpu_mesh["tp"].get_group(),
            )

            # if self.is_engine:
            #     await self.llm.update_weights_from_tensor(
            #         named_tensors=[
            #             (
            #                 name,
            #                 LocalSerializedTensor(values=gathered_serialized_tensors),
            #             )
            #         ],
            #         # load_format=load_format,
            #         flush_cache=index == len(reduce_data)-1,
            #     )
            if self.is_engine:
                await self.llm.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            LocalSerializedTensor(values=gathered_serialized_tensors),
                        )
                    ],
                    # load_format=load_format,
                    flush_cache=index == len(reduce_data)-1,
                )
        torch.cuda.synchronize()

    async def flush_cache(self):
        if self.is_engine:
            await self.llm.flush_cache()
        torch.cuda.synchronize()

    async def offload_weights(self, tags: Optional[List[str]] = None):
        # Currently we only support `weights` and `kv_cache`

        if self.is_engine:
            # avoid offload offloaded param
            tags = self.preprocess_tags(tags, stage="offload")
            if not tags:
                return
            self._logger.info(f"llm_engine.sleep {tags} before: {get_full_proc_memory_info('before llm_engine.sleep')}")
            await self.llm.release_memory_occupation(tags=tags)
            self._logger.info(f"llm_engine.sleep {tags} after: {get_full_proc_memory_info('after llm_engine.sleep')}")
            self.postprocess_tags(tags, stage="offload")
        torch.cuda.synchronize()

    def preprocess_tags(self, tags: Optional[List[str]], stage='onload'):
        """
        preprocess onload, offload tags to avoid duplicate calls
        """
        if tags is None:
            tags = ['kv_cache', 'weights']
        tag_map = {
            "kv_cache": self.kv_cache_onloaded,
            "weights": self.weight_onloaded
        }
        preprocess_tags = []

        for tag in tags:
            onloaded_flag = tag_map[tag]
            if stage == 'onload' and not onloaded_flag:
                preprocess_tags.append(tag)
            elif stage == 'offload' and onloaded_flag:
                preprocess_tags.append(tag)
        return preprocess_tags

    # def postprocess_tags(self, tags: Optional[List[str]], state="onload"):
    #     if state == "onload":
    #         if "kv_cache" in tags:
    #             self.kv_cache_onloaded = True
    #         if "weights" in tags:
    #             self.weight_onloaded = True
    #     elif state == "offload":
    #         if "kv_cache" in tags:
    #             self.kv_cache_onloaded = False
    #         if "weights" in tags:
    #             self.weight_onloaded = False

    def postprocess_tags(self, tags: Optional[List[str]], stage: str = "onload") -> None:
        if tags is None:
            return
        mapping = {
            "kv_cache": "kv_cache_onloaded",
            "weights": "weight_onloaded",
        }
        value = stage == "onload"
        for tag, attr in mapping.items():
            if tag in tags:
                setattr(self, attr, value)

    async def onload_weights(self, tags: Optional[List[str]] = None):
        # Currently we only support `weights` and `kv_cache`
        if self.need_offload:
            await self.offload_weights()
            self.need_offload = False
        if self.is_engine:
            # avoid onload onloaded param
            tags = self.preprocess_tags(tags, stage="onload")
            if not tags:
                return
            self._logger.info(f"llm_engine.wake_up {tags} before: {get_full_proc_memory_info('before llm_engine.wake_up')}")
            await self.llm.resume_memory_occupation(tags=tags)
            self._logger.info(f"llm_engine.wake_up {tags} after: {get_full_proc_memory_info('before llm_engine.wake_up')}")
            self.postprocess_tags(tags, stage="onload")
        torch.cuda.synchronize()

    @property
    def is_engine(self):
        return self.llm and self.llm.tokenizer_manager is not None
