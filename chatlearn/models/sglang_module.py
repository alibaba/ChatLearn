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
from typing import Optional, List, TYPE_CHECKING, Dict
from itertools import batched
import copy

import ray
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from chatlearn.utils.utils import get_full_proc_memory_info
from chatlearn.runtime.decorator import timeit
from chatlearn.utils.mappings import ShardedTensorInfo, build_sharded_info_for_huggingface_model
from .torch_module import TorchModule

try:
    import sglang as sgl
except Exception:
    warnings.warn("SGLang is not installed.")

if TYPE_CHECKING:
    from chatlearn.synchronizer.structs import BucketInfo

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
        tokenizer = AutoTokenizer.from_pretrained(self.module_args['load'], trust_remote_code=True)
        tokenizer.tokenizer = tokenizer
        self.tokenizer = tokenizer

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
            self.llm = sgl.Engine(
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

        # this two flag used for avoid onload,offload twice
        self.kv_cache_onloaded = True
        self.weight_onloaded =True
        self.offload_weights()

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

    def generate(self, query, is_eval):

        outputs = None
        if self.llm and self.llm.tokenizer_manager is not None:
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
                sampling_param_item['max_new_tokens'] = max_tokens
                sampling_params.append(sampling_param_item)

            outputs = self.llm.generate(input_ids=prompts_token_ids,
                                        sampling_params=sampling_params)
        self.flush_cache()
        return outputs

    def update_weights_from_ipc_handles(self, reduce_data):

        # pylint: disable-next=import-outside-toplevel
        from sglang.srt.model_executor.model_runner import LocalSerializedTensor
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

            if self.is_engine:
                self.llm.update_weights_from_tensor(
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

    def flush_cache(self):
        if self.is_engine:
            self.llm.flush_cache()
        torch.cuda.synchronize()

    def offload_weights(self, tags: Optional[List[str]] = None):
        # Currently we only support `weights` and `kv_cache`

        if self.is_engine:
            # avoid offload offloaded param
            if tags is None:
                tags = ['kv_cache', 'weights']
            if not self.kv_cache_onloaded and "kv_cache" in tags:
                tags.pop(tags.index("kv_cache"))
            if not self.weight_onloaded and "weights" in tags:
                tags.pop(tags.index("weights"))
            if not tags:
                return
            self._logger.info(f"llm_engine.sleep {tags} before: {get_full_proc_memory_info('before llm_engine.sleep')}")
            self.llm.release_memory_occupation(tags=tags)
            self._logger.info(f"llm_engine.sleep {tags} after: {get_full_proc_memory_info('after llm_engine.sleep')}")

            if "kv_cache" in tags:
                self.kv_cache_onloaded = False

            if "weights" in tags:
                self.weight_onloaded = False
        torch.cuda.synchronize()

    def onload_weights(self, tags: Optional[List[str]] = None):
        # Currently we only support `weights` and `kv_cache`
        if self.is_engine:
            # avoid onload onloaded param
            if tags is None:
                tags = ['kv_cache', 'weights']
            if self.kv_cache_onloaded and "kv_cache" in tags:
                tags.pop(tags.index("kv_cache"))
            if self.weight_onloaded and "weights" in tags:
                tags.pop(tags.index("weights"))
            if not tags:
                return
            self._logger.info(f"llm_engine.wake_up {tags} before: {get_full_proc_memory_info('before llm_engine.wake_up')}")
            self.llm.resume_memory_occupation(tags=tags)
            self._logger.info(f"llm_engine.wake_up {tags} after: {get_full_proc_memory_info('before llm_engine.wake_up')}")

            if "kv_cache" in tags:
                self.kv_cache_onloaded = True
            if "weights" in tags:
                self.weight_onloaded = True
        torch.cuda.synchronize()


    @property
    def is_engine(self):
        return self.llm and self.llm.tokenizer_manager is not None

    def map_local_param_name_to_global(self):
        model_config = AutoConfig.from_pretrained(self.module_args['load'])
        with torch.device('meta'):
            meta_model = AutoModelForCausalLM.from_config(
                model_config,
                trust_remote_code=True
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
            meta_model = AutoModelForCausalLM.from_config(
                model_config,
                trust_remote_code=True
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
        # pylint: disable-next=import-outside-toplevel
        from sglang.srt.utils import MultiprocessingSerializer
        param_id_to_update = set()
        for bucket in buckets:
            if bucket is None:
                continue
            offset = 0
            if bucket.buffer is None:
                raise ValueError("Attempt to read from a bucket without buffer")
            for offset, sharded_tensor_info in bucket.recv_layout:
                if sharded_tensor_info.param_id not in param_id_to_update:
                    param_id_to_update.add(sharded_tensor_info.param_id)

        # TODO: Since SGLang v0.5.0rc1, we can use _update_weights_from_flattened_bucket
        # to reduce num of IPC handles. Fix later
        # NOTE: Chunk to (CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO // 2) weights per-call
        # see: https://github.com/pytorch/pytorch/blob/d8d589bd3ac7a426ca21377f3d5c318e5e8ac055/torch/csrc/CudaIPCTypes.cpp#L100
        for param_ids in batched(param_id_to_update, n=500):
            params_to_update: Dict[str, torch.Tensor] = {}
            for bucket in buckets:
                if bucket is None:
                    continue
                offset = 0
                for offset, sharded_tensor_info in bucket.recv_layout:
                    param_id = sharded_tensor_info.param_id
                    if param_id not in param_ids:
                        continue
                    param_name = self.param_id_to_local_name[param_id]
                    if param_name not in params_to_update:
                        shard_info = self.param_id_to_metadata[param_id]
                        params_to_update[param_name] = torch.empty(
                            shard_info.global_shape,
                            dtype=shard_info.dtype,
                            device='cuda'
                        )
                    weight = params_to_update[param_name]
                    shard = sharded_tensor_info.index(weight)
                    comm_dtype = sharded_tensor_info.dtype
                    numel = shard.numel()
                    byte_data = bucket.buffer[offset: offset + numel * comm_dtype.itemsize]
                    # NOTE: if shard.dtype != comm_dtype, an implicit datatype conversion will happen
                    shard.copy_(byte_data.view(comm_dtype).view(shard.shape))
                    offset += numel * comm_dtype.itemsize
            for name, weight in params_to_update.items():
                params_to_update[name] = MultiprocessingSerializer.serialize(weight)
            self.update_weights_from_ipc_handles(params_to_update)
            del params_to_update

        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
