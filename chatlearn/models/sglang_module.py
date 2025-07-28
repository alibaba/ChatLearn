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
import os
import math

import ray
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer
import sglang as sgl

from .torch_module import TorchModule

class SGLangModule(TorchModule):

    def __init__(self, name: str, args=None, replica_id: int=0, **kwargs):
        """The chatlearn wrapper for a sglang model.
        """

        super().__init__(name, args=args, replica_id=replica_id)

        assert self.total_gpu > 0, "SGLang requires at least one GPU"
        assert not self.trainable, "SGLang does not support training"
        # TODO: support expert-model parallel, 
        assert self.module_args.expert_model_parallel_size == 1, "Expert Parallel of SGLang is not supported"
        assert self.module_args.pipeline_model_parallel_size == 1, "Pipeline Parallel of SGLang is not supported"

        self._num_gpu_per_replica = (
            self.module_args.tensor_model_parallel_size
        )
        self.tensor_model_parallel_size = self.module_args.tensor_model_parallel_size

        assert self.total_gpu % self._num_gpu_per_replica == 0, \
            "The GPUs assigned to this model must be divisible by num_gpu_per_replica"

        self._num_replica = self.total_gpu // self._num_gpu_per_replica

        # get gpu_per_node used for setup sglang
        resource = ray.nodes()[0]['Resources']
        self.gpu_per_node = int(resource['GPU'])

        self._metric_prefix = 'sglang_inference'


    def init(self):
        """
        initialize distributed env
        """
        # we need cpu process group for communicate ipc handle
        dist.init_process_group(backend=f"cpu:gloo,cuda:nccl")

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

    def setup(self):
        super().setup()
        tokenizer = AutoTokenizer.from_pretrained(self.module_args['load'], trust_remote_code=True)
        tokenizer.tokenizer = tokenizer
        self.tokenizer = tokenizer
    
    def setup_sglang(self):
        
        nnodes_per_replica = math.ceil(self.tensor_model_parallel_size / self.gpu_per_node)
        if nnodes_per_replica > 1:
            dist_init_addr = f"{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}"
        else:
            dist_init_addr = None
        
        load_format = self.module_args.get("vllm_load_format", "dummy")
        tp_size_per_node = self._tp_size // nnodes_per_replica
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        dtype = self.module_args.get("dtype", "bfloat16")

        if first_rank_in_node:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self.llm = sgl.Engine(
                model_path=self.module_args['load'],
                dtype=dtype,
                mem_fraction_static=self.config.gpu_memory_utilization,
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=True,

                port=40000 + self.replica_id,
                # NOTE(Chenyang): if you want to debug the SGLang engine output
                # please set the following parameters
                # Otherwise, it will make the engine run too slow
                # log_level="INFO",
                # log_requests=True,
                # log_requests_level=2,
                # max_running_requests=1,

                # mm_attention_backend="fa3",
                # attention_backend="fa3",
                # In async mode, we want token in token out.
                # skip_tokenizer_init=self.config.mode == "async",
            )
        else:
            self._engine = None
        dist.barriar()
        breakpoint()
        


    def generate(self, query):
        pass