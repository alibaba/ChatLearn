# pylint: disable=invalid-overridden-method,abstract-method,arguments-differ,import-outside-toplevel,unused-argument
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
import copy
import math
import multiprocessing as mp
import os
import traceback
import warnings
from unittest.mock import MagicMock
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any
from collections import defaultdict

import ray
from ray import ObjectRef
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM, AutoConfig, AutoProcessor

from chatlearn.runtime.decorator import timeit, compute_decorator
from chatlearn.utils.utils import get_full_proc_memory_info
from chatlearn.utils.mappings import ShardedTensorInfo
from chatlearn.utils.mappings.huggingface_helpers import build_sharded_info_for_huggingface_model

from .torch_module import TorchModule

if TYPE_CHECKING:
    from chatlearn.synchronizer.structs import BucketInfo

try:
    import sglang
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.managers.io_struct import (
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
        UpdateWeightsFromTensorReqInput,
    )
    from sglang.srt.utils import (
        MultiprocessingSerializer,
        assert_pkg_version,
        get_bool_env_var,
        is_cuda,
        set_prometheus_multiproc_dir,
        set_ulimit,
    )
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorMetadata

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
        # ChatLearn modify part

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
    # Mock Engine
    Engine = MagicMock()


def metric_collect(rets, max_response_tokens_length):
    # collect metric
    response_token_length = [ret["response_token_length"] for ret in rets]
    prompt_token_length = [ret["prompt_token_length"] for ret in rets]
    clip_ratio = sum(
        ret["response_token_length"] >= ret.get("max_generate_token_length", max_response_tokens_length) \
            for ret in rets
    ) / len(rets)
    response_token_length.sort()
    inference_stats = {
        "response_token_length": sum(response_token_length)
        / len(response_token_length),
        "prompt_token_length": sum(prompt_token_length) / len(prompt_token_length),
        "response_clip_ratio": clip_ratio,
        "response_max": max(response_token_length),
        "response_25_percentile": np.percentile(response_token_length, 25),
        "response_50_percentile": np.percentile(response_token_length, 50),
        "response_75_percentile": np.percentile(response_token_length, 75),
    }
    return inference_stats


# modified from https://github.com/volcengine/verl/blob/main/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L128
class AsyncEngine(Engine):
    """
    patch for sglang engine, because chatCompletion is an async method, it makes the whole ray actor be an async actor
    which can not call loop.run_until_complete. So we need to make the engine to be an async class
    """

    async def release_memory_occupation(self, tags: Optional[list[str]] = None):
        """Release GPU occupation temporarily."""
        if tags is None:
            obj = ReleaseMemoryOccupationReqInput()
        else:
            obj = ReleaseMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None):
        """Resume GPU occupation."""
        if tags is None:
            obj = ResumeMemoryOccupationReqInput()
        else:
            obj = ResumeMemoryOccupationReqInput(tags=tags)

        return await self.tokenizer_manager.resume_memory_occupation(obj, None)

    async def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be false
        to avoid duplicated cache cleaning operation."""
        if load_format == "flattened_bucket":
            serialized_named_tensors = named_tensors
        else:
            serialized_named_tensors = [
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(self.server_args.tp_size)
            ]
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
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

    def __init__(self, name: str, args=None, replica_id: int = 0):
        """The chatlearn wrapper for a sglang model."""
        super().__init__(name, args=args, replica_id=replica_id)
        self.tensor_model_parallel_size = self.module_args.tensor_model_parallel_size
        # get gpu_per_node used for setup sglang
        resource = ray.nodes()[0]["Resources"]
        self.gpu_per_node = int(resource["GPU"])
        self.llm = None
        self._metric_prefix = "rollout"

    def init(self):
        """
        initialize distributed env
        """
        # we need cpu process group for communicate ipc handle
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")

        # init cpu_mesh
        self.cpu_mesh = init_device_mesh(
            device_type="cpu",
            mesh_shape=(self.tensor_model_parallel_size,),
            mesh_dim_names=["tp"],
        )
        self._tp_rank = self.cpu_mesh["tp"].get_local_rank()
        self._tp_size = self.cpu_mesh["tp"].size()
        torch.cuda.get_device_capability()

        visible_devices = [None] * self.cpu_mesh.size()

        torch.distributed.all_gather_object(
            visible_devices,
            os.environ["CUDA_VISIBLE_DEVICES"],
            self.cpu_mesh.get_group("tp"),
        )
        self.visible_devices_set = set(",".join(visible_devices).split(","))

        # used for init sglang engine in ray actor
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            sorted(list(self.visible_devices_set))
        )
        dist.barrier(group=self.cpu_mesh.get_group())

    @timeit()
    def setup_engine(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.module_args.load, trust_remote_code=self.module_args.trust_remote_code
        )

        if self.runtime_args.model_type == 'vlm':
            # processor is needed for qwenvl
            self.processor = AutoProcessor.from_pretrained(self.module_args.load, trust_remote_code=self.module_args.trust_remote_code)
        else:
            self.processor = None

        if self.llm is not None:  # for evaluator not setup twice
            dist.barrier()
            return
        nnodes_per_replica = math.ceil(
            self.tensor_model_parallel_size / self.gpu_per_node
        )
        if nnodes_per_replica > 1:
            dist_init_addr = (
                f"{os.environ['MASTER_ADDR']}:{os.environ['SGLANG_NCCL_PORT']}"
            )
        else:
            dist_init_addr = None

        load_format = self.module_args.get("load_format", "dummy")
        tp_size_per_node = self._tp_size // nnodes_per_replica
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        dtype = self.module_args.get("dtype", "bfloat16")

        if first_rank_in_node:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            # avoid ray actor become an async actor in sync mode
            llm_cls = sglang.Engine if self.module_args.is_sync_mode else AsyncEngine
            self.llm = llm_cls(
                model_path=self.module_args["load"],
                dtype=dtype,
                mem_fraction_static=self.module_args.get(
                    "gpu_memory_utilization", 0.85
                ),
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes_per_replica,
                trust_remote_code=self.module_args.trust_remote_code,
                port=40000 + self.replica_id,
                nccl_port=int(os.environ["SGLANG_NCCL_PORT"]),
                mm_attention_backend="fa3",
                attention_backend="fa3",
                skip_tokenizer_init=True,
                disable_cuda_graph=self.module_args.get("enforce_eager", False),
            )

        # this two flag used for avoid onload,offload twice
        self.kv_cache_onloaded = True
        self.weight_onloaded = True

        self.need_offload = True
        dist.barrier()

    def _get_sampling_params(self, is_eval):
        temperature = 0.0
        if not self.module_args.get("use_beam_search", False):
            temperature = (
                self.module_args.get("eval_temperature", 1.0)
                if is_eval
                else self.module_args.get("temperature", 1.0)
            )
        top_p = (
            self.module_args.get("eval_top_p", 1.0)
            if is_eval
            else self.module_args.get("top_p", 1.0)
        )
        top_k = (
            self.module_args.get("eval_top_k", -1)
            if is_eval
            else self.module_args.get("top_k", -1)
        )
        min_p = (
            self.module_args.get("eval_min_p", 0.0)
            if is_eval
            else self.module_args.get("min_p", 0.0)
        )
        presence_penalty = (
            self.module_args.get("eval_presence_penalty", 0.0)
            if is_eval
            else self.module_args.get("presence_penalty", 0.0)
        )
        frequency_penalty = (
            self.module_args.get("eval_frequency_penalty", 0.0)
            if is_eval
            else self.module_args.get("frequency_penalty", 0.0)
        )
        repetition_penalty = (
            self.module_args.get("eval_repetition_penalty", 1.0)
            if is_eval
            else self.module_args.get("repetition_penalty", 1.0)
        )
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
            "skip_special_tokens": self.module_args.get("skip_special_tokens", True),
        }

        return sampling_params

    def preprocess_data(self, query: List[Dict], is_eval: bool):
        """
        generate sampling parameter query-wise
        """
        max_response_tokens_length = self.module_args.max_response_tokens_length
        prompts_token_ids = [q["input_ids"] for q in query]

        if self.runtime_args.model_type == 'vlm':
            # vlm
            image_data = [q["multi_modal_data"]["image"] for q in query]
        else:
            # llm
            image_data = None

        sampling_param = self._get_sampling_params(is_eval)
        sampling_params = []

        for q in query:
            # When partial_rollout is enabled, max_generate_token_length will be set by RolloutManager
            # for different rollotu rounds.
            # When partial_rollout is disabled, max_response_tokens_length from config will be used
            max_tokens = q.get("max_generate_token_length", max_response_tokens_length)
            sampling_param_item = copy.deepcopy(sampling_param)
            sampling_param_item["max_new_tokens"] = max_tokens
            sampling_params.append(sampling_param_item)

        return prompts_token_ids, sampling_params, image_data

    def generate(self, query: List[Dict], is_eval: bool) -> List[Dict]:
        outputs = None
        if self.is_engine():
            prompts_token_ids, sampling_params, image_data = self.preprocess_data(query, is_eval)
            outputs = self.llm.generate(
                input_ids=prompts_token_ids,
                sampling_params=sampling_params,
                image_data=image_data
            )
        self.flush_cache()
        return outputs

    def dump_parameters(self, dump_path_root):
        os.makedirs(dump_path_root, exist_ok=True)
        self.onload()
        self.llm.save_sharded_model(path=dump_path_root, pattern=None, max_size=None)
        self.offload()

    def update_weights_from_ipc_handles(self, reduce_data):
        gathered_data = None
        if self.is_engine():
            gathered_data = [None] * self._tp_size
        dist.gather_object(
            obj=reduce_data,
            object_gather_list=gathered_data,
            dst=self.cpu_mesh["tp"].mesh.tolist()[0],
            group=self.cpu_mesh["tp"].get_group(),
        )
        if self.is_engine():
            self.llm.update_weights_from_tensor(
                named_tensors=gathered_data,
                load_format="flattened_bucket",
            )
        torch.cuda.synchronize()

    def flush_cache(self):
        if self.is_engine():
            self.llm.flush_cache()
        torch.cuda.synchronize()

    def preprocess_tags(self, tags: Optional[List[str]], stage="onload"):
        """
        preprocess onload, offload tags to avoid duplicate calls
        """
        if tags is None:
            tags = ["kv_cache", "weights"]
        tag_map = {"kv_cache": self.kv_cache_onloaded, "weights": self.weight_onloaded}
        preprocess_tags = []

        for tag in tags:
            onloaded_flag = tag_map[tag]
            if stage == "onload" and not onloaded_flag:
                preprocess_tags.append(tag)
            elif stage == "offload" and onloaded_flag:
                preprocess_tags.append(tag)
        return preprocess_tags

    def postprocess_tags(
        self, tags: Optional[List[str]], stage: str = "onload"
    ) -> None:
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

    @timeit()
    def offload(self, tags: Optional[List[str]] = None):
        # Currently we only support `weights` and `kv_cache`
        if self.is_engine():
            # avoid offload offloaded param
            tags = self.preprocess_tags(tags, stage="offload")
            if not tags:
                return
            self._logger.info(
                f"llm_engine.sleep {tags} before: {get_full_proc_memory_info('before llm_engine.sleep')}"
            )
            self.llm.release_memory_occupation(tags=tags)
            self._logger.info(
                f"llm_engine.sleep {tags} after: {get_full_proc_memory_info('after llm_engine.sleep')}"
            )
            self.postprocess_tags(tags, stage="offload")
        torch.cuda.synchronize()

    @timeit()
    def onload(self, tags: Optional[List[str]] = None):
        # Currently we only support `weights` and `kv_cache`
        if self.need_offload:
            self.offload()
            self.need_offload = False
        if self.is_engine():
            # avoid onload onloaded param
            tags = self.preprocess_tags(tags, stage="onload")
            if not tags:
                return
            self._logger.info(
                f"llm_engine.wake_up {tags} before: {get_full_proc_memory_info('before llm_engine.wake_up')}"
            )
            self.llm.resume_memory_occupation(tags=tags)
            self._logger.info(
                f"llm_engine.wake_up {tags} after: {get_full_proc_memory_info('before llm_engine.wake_up')}"
            )
            self.postprocess_tags(tags, stage="onload")
        torch.cuda.synchronize()

    def is_engine(self):
        return self.llm and self.llm.tokenizer_manager is not None

    def map_local_param_name_to_global(self) -> List[str]:
        """Map names of weights on each rank to a unique name.
        
        Returns:
            List[str]: A list of unique global names for each weight 
        on this rank.
        """
        model_config = AutoConfig.from_pretrained(self.module_args['load'])

        with torch.device('meta'):
            if self.runtime_args.model_type == 'vlm':
                meta_model = AutoModelForImageTextToText.from_config(
                    model_config,
                    trust_remote_code=self.module_args.trust_remote_code
                )
            else:
                meta_model = AutoModelForCausalLM.from_config(
                    model_config,
                    trust_remote_code=self.module_args.trust_remote_code
                )
        names = list(meta_model.state_dict().keys())
        self.global_name_to_local_name = {n: n for n in names}
        return names

    @torch.no_grad()
    def get_parameter_metadata(self) -> Dict[str, ShardedTensorInfo]:
        """Collect parameter shape info of this rank
        """
        if self.local_name_to_param_id is None:
            raise ValueError("Call set_param_id before call this function")

        model_config = AutoConfig.from_pretrained(self.module_args['load'])

        with torch.device('meta'):
            if self.runtime_args.model_type == 'vlm':
                meta_model = AutoModelForImageTextToText.from_config(
                    model_config,
                    trust_remote_code=self.module_args.trust_remote_code
                )
            else:
                meta_model = AutoModelForCausalLM.from_config(
                    model_config,
                    trust_remote_code=self.module_args.trust_remote_code
                )
        infos = {}
        for name, sharded_info in build_sharded_info_for_huggingface_model(meta_model).items():
            param_id = self.local_name_to_param_id[name]
            sharded_info.param_id = param_id
            infos[param_id] = sharded_info
        return infos

    def parameter_sync(self):
        """Perform parameter synchronization on this worker."""
        if self.synchronizer is None:
            raise ValueError("Synchronizer is not initialized.")
        self.param_id_to_metadata = self.get_parameter_metadata()
        self.synchronizer.parameter_sync()
        self.param_id_to_metadata = None

    @torch.no_grad()
    def update_weights_from_buckets(self, buckets: List[Optional['BucketInfo']]):
        """Used for Mcore2SGLang Parameter Sync
        """
        from sglang.srt.patch_torch import monkey_patch_torch_reductions
        monkey_patch_torch_reductions()
        param_id_to_update = set()
        for bucket in buckets:
            if bucket is None:
                continue
            if bucket.buffer is None:
                raise ValueError("Attempt to read from a bucket without buffer")
            param_id_to_update.update({sharded_tensor_info.param_id for _, sharded_tensor_info in bucket.recv_layout})

        param_id_to_bucket = defaultdict(list)
        for bucket_idx, bucket in enumerate(buckets):
            if bucket is None:
                continue
            for shard_idx, (offset, sharded_tensor_info) in enumerate(bucket.recv_layout):
                param_id_to_bucket[sharded_tensor_info.param_id].append((bucket_idx, shard_idx))

        # 1-dim concated flattened tensor
        buffer = None
        buffer_offset = 0
        buffer_size = 4 * 1024 ** 3
        # metadata: name, shape, dtype, start_idx, end_idx, numel for every tensor item in buffer
        metadatas: List[FlattenedTensorMetadata] = []
        for param_id in param_id_to_update:
            param_name = self.param_id_to_local_name[param_id]
            shard_info = self.param_id_to_metadata[param_id]

            if self.runtime_args.model_type == 'vlm':
                if 'visual' in param_name:
                    param_name = param_name.replace("model.", "")
                else:
                    param_name = param_name.replace("model.language_model.", "model.")

            if buffer is None:
                buffer = torch.empty(buffer_size, dtype=shard_info.dtype, device='cuda')
                buffer_offset = 0
                metadatas = []
            elif buffer.dtype != shard_info.dtype or buffer_offset + shard_info.numel() > buffer_size:
                bucket_dict = {"flattened_tensor": buffer[:buffer_offset], "metadata": metadatas}
                serialized_bucket = MultiprocessingSerializer.serialize(
                    bucket_dict, output_str=True
                )
                self.update_weights_from_ipc_handles(serialized_bucket)
                buffer = torch.empty(buffer_size, dtype=shard_info.dtype, device='cuda')
                buffer_offset = 0
                metadatas = []

            weight = buffer[buffer_offset: buffer_offset + shard_info.numel()].view(shard_info.global_shape)
            metadatas.append(FlattenedTensorMetadata(
                name=param_name,
                shape=weight.shape,
                dtype=weight.dtype,
                start_idx=buffer_offset,
                end_idx=buffer_offset + shard_info.numel(),
                numel=shard_info.numel(),
            ))
            for bucket_idx, shard_idx in param_id_to_bucket[param_id]:
                bucket = buckets[bucket_idx]
                offset, sharded_tensor_info = bucket.recv_layout[shard_idx]
                byte_data = bucket.buffer[offset: offset + sharded_tensor_info.size]
                shard = sharded_tensor_info.index(weight)
                comm_dtype = sharded_tensor_info.dtype
                # NOTE: if shard.dtype != comm_dtype, an implicit datatype conversion will happen
                shard.copy_(byte_data.view(comm_dtype).view(shard.shape))

            buffer_offset += shard_info.numel()

        if buffer_offset > 0:
            bucket_dict = {"flattened_tensor": buffer[:buffer_offset], "metadata": metadatas}
            serialized_bucket = MultiprocessingSerializer.serialize(
                bucket_dict, output_str=True
            )
            self.update_weights_from_ipc_handles(serialized_bucket)

        del buffer, weight, shard, bucket_dict
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    def postprocess_func(
        self,
        batched_outputs: List[Dict[str, Any]],
        input_data_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if isinstance(batched_outputs[0], ObjectRef):
            batched_outputs = ray.get(batched_outputs)
        data_output = []
        for output, input_data in zip(batched_outputs, input_data_list):
            prompt_token_ids = input_data["input_ids"]
            output_tokens = output["output_ids"]
            response_token_length = output["meta_info"]["completion_tokens"]
            str_outputs = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            all_tokens = torch.tensor(prompt_token_ids + output_tokens)
            input_data.update(
                {
                    "all_tokens": all_tokens,
                    "response_token_length": response_token_length,
                    "str_outputs": str_outputs,
                    "all_token_length": len(prompt_token_ids) + len(output_tokens)
                }
            )
            if "rollout_round" in input_data:
                input_data["rollout_round"] += 1
            data_output.append(input_data)

        print("str_outputs", data_output[0]["str_outputs"])
        print("data_sources", data_output[0]["data_source"])
        print("ground_truth", data_output[0]["ground_truth"])
        return data_output

    @compute_decorator(trainable=False, rollout=True)
    @timeit()
    def eval_forward(self, data, iteration=0, **kwargs):
        return self._forward_step(data, iteration, True)

    def _forward_step(
        self, data, iteration, is_eval
    ):
        outputs = self.generate(data, is_eval)

        if outputs is not None:
            rets = self.postprocess_func(outputs, data)
            return rets

    @compute_decorator(trainable=False, rollout=True)
    @timeit()
    def forward_step(
        self, data: List[Dict[str, Any]], iteration=0, **kwargs
    ) -> List[Dict[str, Any]]:

        rets = self._forward_step(data, iteration, False)
        # collect metric
        self._metric_list.append(metric_collect(rets, self.module_args.max_response_tokens_length))
        return rets


class AsyncSGLangModule(SGLangModule):
    """AsyncSGLangModule"""

    def __init__(self, name: str, args=None, replica_id: int = 0):
        """The chatlearn wrapper for a async sglang model."""
        super().__init__(name, args=args, replica_id=replica_id)

    async def generate(self, query: List[Dict], is_eval: bool) -> List[Dict]:
        outputs = None
        if self.is_engine():
            prompts_token_ids, sampling_params = self.preprocess_data(query, is_eval)
            outputs = await self.llm.async_generate(
                prompt=None,
                sampling_params=sampling_params,
                return_logprob=False, # sglang has memory leaky problem while return_logprob=True
                input_ids=prompts_token_ids,
            )
        return outputs

    async def dump_parameters(self, dump_path_root):
        os.makedirs(dump_path_root, exist_ok=True)
        await self.onload()
        self.llm.save_sharded_model(path=dump_path_root, pattern=None, max_size=None)
        await self.offload()

    async def generate_per_request(self, query: Dict, is_eval: bool) -> Dict:
        outputs = None
        if self.is_engine():
            prompts_token_ids = query['input_ids']
            if self.runtime_args.model_type == 'vlm':
                # vlm
                image_data = query["multi_modal_data"]["image"]
            else:
                # llm
                image_data = None
            sampling_param = self._get_sampling_params(is_eval)
            sampling_param["max_new_tokens"] = self.module_args.max_response_tokens_length
            outputs = await self.llm.async_generate(
                prompt=None,
                sampling_params=sampling_param,
                return_logprob=False,
                input_ids=prompts_token_ids,
                image_data=image_data
            )
        return outputs

    async def update_weights_from_ipc_handles(self, reduce_data):

        gathered_data = None
        if self.is_engine():
            gathered_data = [None] * self._tp_size
        dist.gather_object(
            obj=reduce_data,
            object_gather_list=gathered_data,
            dst=self.cpu_mesh["tp"].mesh.tolist()[0],
            group=self.cpu_mesh["tp"].get_group(),
        )
        if self.is_engine():
            await self.llm.update_weights_from_tensor(
                named_tensors=gathered_data,
                load_format="flattened_bucket",
            )
        torch.cuda.synchronize()

    @torch.no_grad()
    async def update_weights_from_buckets(self, buckets: List[Optional['BucketInfo']]):
        from sglang.srt.patch_torch import monkey_patch_torch_reductions
        monkey_patch_torch_reductions()
        param_id_to_update = set()
        for bucket in buckets:
            if bucket is None:
                continue
            if bucket.buffer is None:
                raise ValueError("Attempt to read from a bucket without buffer")
            param_id_to_update.update({sharded_tensor_info.param_id for _, sharded_tensor_info in bucket.recv_layout})

        param_id_to_bucket = defaultdict(list)
        for bucket_idx, bucket in enumerate(buckets):
            if bucket is None:
                continue
            for shard_idx, (offset, sharded_tensor_info) in enumerate(bucket.recv_layout):
                param_id_to_bucket[sharded_tensor_info.param_id].append((bucket_idx, shard_idx))

        buffer = None
        buffer_offset = 0
        buffer_size = 4 * 1024 ** 3
        metadatas = []
        for param_id in param_id_to_update:
            param_name = self.param_id_to_local_name[param_id]
            shard_info = self.param_id_to_metadata[param_id]

            if self.runtime_args.model_type == 'vlm':
                if 'visual' in param_name:
                    param_name = param_name.replace("model.", "")
                else:
                    param_name = param_name.replace("model.language_model.", "model.")

            if buffer is None:
                buffer = torch.empty(buffer_size, dtype=shard_info.dtype, device='cuda')
                buffer_offset = 0
                metadatas = []
            elif buffer.dtype != shard_info.dtype or buffer_offset + shard_info.numel() > buffer_size:
                bucket_dict = {"flattened_tensor": buffer[:buffer_offset], "metadata": metadatas}
                serialized_bucket = MultiprocessingSerializer.serialize(
                    bucket_dict, output_str=True
                )
                await self.update_weights_from_ipc_handles(serialized_bucket)
                buffer = torch.empty(buffer_size, dtype=shard_info.dtype, device='cuda')
                buffer_offset = 0
                metadatas = []

            weight = buffer[buffer_offset: buffer_offset + shard_info.numel()].view(shard_info.global_shape)
            metadatas.append(FlattenedTensorMetadata(
                name=param_name,
                shape=weight.shape,
                dtype=weight.dtype,
                start_idx=buffer_offset,
                end_idx=buffer_offset + shard_info.numel(),
                numel=shard_info.numel(),
            ))
            for bucket_idx, shard_idx in param_id_to_bucket[param_id]:
                bucket = buckets[bucket_idx]
                offset, sharded_tensor_info = bucket.recv_layout[shard_idx]
                byte_data = bucket.buffer[offset: offset + sharded_tensor_info.size]
                shard = sharded_tensor_info.index(weight)
                comm_dtype = sharded_tensor_info.dtype
                # NOTE: if shard.dtype != comm_dtype, an implicit datatype conversion will happen
                shard.copy_(byte_data.view(comm_dtype).view(shard.shape))

            buffer_offset += shard_info.numel()

        if buffer_offset > 0:
            bucket_dict = {"flattened_tensor": buffer[:buffer_offset], "metadata": metadatas}
            serialized_bucket = MultiprocessingSerializer.serialize(
                bucket_dict, output_str=True
            )
            await self.update_weights_from_ipc_handles(serialized_bucket)

        del buffer, weight, shard, bucket_dict
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    async def flush_cache(self):
        if self.is_engine():
            await self.llm.flush_cache()
        torch.cuda.synchronize()

    @timeit()
    async def offload(self, tags: Optional[List[str]] = None):
        # Currently we only support `weights` and `kv_cache`

        if self.is_engine():
            # avoid offload offloaded param
            tags = self.preprocess_tags(tags, stage="offload")
            if not tags:
                return
            self._logger.info(
                f"llm_engine.sleep {tags} before: {get_full_proc_memory_info('before llm_engine.sleep')}"
            )
            await self.llm.release_memory_occupation(tags=tags)
            self._logger.info(
                f"llm_engine.sleep {tags} after: {get_full_proc_memory_info('after llm_engine.sleep')}"
            )
            self.postprocess_tags(tags, stage="offload")
        torch.cuda.synchronize()

    @timeit()
    async def onload(self, tags: Optional[List[str]] = None):
        # Currently we only support `weights` and `kv_cache`
        if self.need_offload:
            await self.offload()
            self.need_offload = False
        if self.is_engine():
            # avoid onload onloaded param
            tags = self.preprocess_tags(tags, stage="onload")
            if not tags:
                return
            self._logger.info(
                f"llm_engine.wake_up {tags} before: {get_full_proc_memory_info('before llm_engine.wake_up')}"
            )
            await self.llm.resume_memory_occupation(tags=tags)
            self._logger.info(
                f"llm_engine.wake_up {tags} after: {get_full_proc_memory_info('before llm_engine.wake_up')}"
            )
            self.postprocess_tags(tags, stage="onload")
        torch.cuda.synchronize()

    @compute_decorator(trainable=False, rollout=True)
    @timeit()
    async def eval_forward(self, data, iteration=0, **kwargs):
        return await self._forward_step(data, iteration, True)

    async def _forward_step(
        self, data, iteration, is_eval
    ):  # pylint: disable=unused-argument
        outputs = await self.generate(data, is_eval)

        if outputs is not None:
            rets = self.postprocess_func(outputs, data)
            return rets

    @compute_decorator(trainable=False, rollout=True)
    @timeit()
    async def forward_step(
        self, data: List[Dict[str, Any]], iteration=0, **kwargs
    ) -> List[Dict[str, Any]]:  # pylint: disable=unused-argument
        rets = await self._forward_step(data, iteration, False)
        # collect metric
        self._metric_list.append(metric_collect(rets, self.module_args.max_response_tokens_length))
        return rets

    async def parameter_sync(self):
        """Perform parameter synchronization on this worker."""
        if self.synchronizer is None:
            raise ValueError("Synchronizer is not initialized.")
        self.param_id_to_metadata = self.get_parameter_metadata()
        await self.synchronizer.async_parameter_sync()
        self.param_id_to_metadata = None
