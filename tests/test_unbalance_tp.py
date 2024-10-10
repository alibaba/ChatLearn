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
"""Test when trainer_tp < inference_tp but trainer_tp can divide inference_tp."""

import os
import ray
import time

import torch
from collections import defaultdict
from tqdm import tqdm

import chatlearn
from chatlearn import TorchModule
from chatlearn.runtime.engine import Engine
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.trainer import Trainer
from chatlearn.utils import future


trainer_params = {}
inference_params = {}

ParamsToSync_Inference = {
    "weight_1" : [8, 8],
    "bias_1" : [8],
    "weight_2" : [8, 10],
    "bias_2": [10]
}

ParamsToSync_Trainer = {
    0 : {
        "weight_1" : [16, 8],
        "bias_1" : [8],
    },
    1 : {
        "weight_2" : [16, 10],
        "bias_2": [10]
    } 
}

class TestTorchModule(TorchModule):
    def get_local_param_ranks(self):
        return [self._get_rank()], 0

    def tensor_parallel_rank(self):
        return int(os.environ["RANK"])

    def pipeline_parallel_rank(self):
        return self.tensor_parallel_rank() // self.tensor_model_parallel_size()

    def check_param_exists(self, names):
        return True


class CustomEngine(Engine):
    """Custom Engine"""

    def __init__(self,
                 policy: TestTorchModule,
                 policy_trainer: TestTorchModule):
        def env_compute_flow(batch):
            policy_out = policy.forward_step(batch)
            return policy_out

        def trainer_compute_flow(batch):
            policy_trainer.train_step(batch)

        env = Environment(env_compute_flow)
        trainer = Trainer(trainer_compute_flow)
        super().__init__(env, trainer)
        self.set_parameter_sync(policy_trainer, policy)


class PolicyModel(TestTorchModule):

    def get_parameter_names(self, requires_grad=True):
        return list(ParamsToSync_Inference.keys())

    def get_parameter_shape(self, names):
        key = 0 if self.tensor_parallel_rank() < 4 else 1
        shape = []
        for name in names:
            shape.append((name, torch.Size(ParamsToSync_Inference[name])))
        return shape

    def get_parameter(self, name):
        return inference_params[f"{self.tensor_parallel_rank()}_{self.pipeline_parallel_rank()}"][name]

    def set_sync_parameters(self, trainable_param_names, pipe_stage=0, parameters_to_sync=None):
        if parameters_to_sync is None:
            parameters_to_sync = self._parameters_to_sync
        all_params = []
        tmp = {}
        for name, shape in ParamsToSync_Inference.items():
            tensor = torch.rand(shape).cuda()
            tmp[name] = tensor
            all_params.append((name, tensor))
        global inference_params
        inference_params[f"{self.tensor_parallel_rank()}_{self.pipeline_parallel_rank()}"] = tmp
        key = self.tensor_parallel_rank() % 2
        start = key * 2
        end = start + 2
        parameters_to_sync[pipe_stage] = all_params[start:end]

    def set_recv_parameters(self, rank, trainable_param_names, pipe_stage=0):
        """
        :meta private:
        """
        all_params = []
        global inference_params
        for name, shape in ParamsToSync_Inference.items():
            tensor = inference_params[f"{self.tensor_parallel_rank()}_{self.pipeline_parallel_rank()}"][name]
            all_params.append((name, tensor))
        key = (self.tensor_parallel_rank() + 1) % 2
        start = key * 2
        end = start + 2
        parameters_to_recv = defaultdict(list)
        parameters_to_recv[pipe_stage] = all_params[start:end]
        self._parameters_to_recv[rank] = parameters_to_recv

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        query = data["query"]
        bs = query.size(0)
        data["policy_out"] = torch.ones([bs, 1024]).cuda()
        return data

    @property
    def data_parallel_size(self):
        return 8 // (self.tensor_model_parallel_size() * self.pipeline_model_parallel_size())

    @property
    def data_parallel_rank(self):
        return self.tensor_parallel_rank() // (self.tensor_model_parallel_size() * self.pipeline_model_parallel_size())


class PPOPolicy(TestTorchModule):

    def build_pipeline_layer_name_mapping(self, num_target_pipe_stage, target_pipe_rank, tgt_layer_offset, requires_grad=True):
        src_names = ParamsToSync_Trainer[self.pipeline_parallel_rank()].keys()
        dst_src_mappings = {}
        for key, value in zip(src_names, src_names):
            dst_src_mappings[key] = value
        return dst_src_mappings

    def get_parameter_shape(self, names):
        shape = []
        for name in names:
            shape.append((name, torch.Size(ParamsToSync_Trainer[self.pipeline_parallel_rank()][name])))
        return shape

    def get_parameter(self, name):
        return trainer_params[f"{self.tensor_parallel_rank()}_{self.pipeline_parallel_rank()}"][name]

    def set_sync_parameters(self, trainable_param_names, pipe_stage=0, parameters_to_sync=None):
        if parameters_to_sync is None:
            parameters_to_sync = self._parameters_to_sync
        tmp = {}
        for name, shape in ParamsToSync_Trainer[self.pipeline_parallel_rank()].items():
            tensor = torch.rand(shape).cuda()
            tmp[name] = tensor
            parameters_to_sync[pipe_stage].append((name, tensor))
        global trainer_params
        trainer_params[f"{self.tensor_parallel_rank()}_{self.pipeline_parallel_rank()}"] = tmp

    def _get_rank(self):
        return int(os.environ["RANK"])

    @property
    def data_parallel_size(self):
        return 8 // (self.tensor_model_parallel_size() * self.pipeline_model_parallel_size())

    @property
    def data_parallel_rank(self):
        return self._get_rank() // (self.tensor_model_parallel_size() * self.pipeline_model_parallel_size())


# tuples: (dst_tp, src_pp, src_tp)
tuples = (8, 2, 4)

chatlearn.init()
for _, model_config in chatlearn.get_args().models.items():
    model_config.num_gpu = 8
chatlearn.get_args().models['policy'].tensor_model_parallel_size = tuples[0]
chatlearn.get_args().models['ppo_policy'].pipeline_model_parallel_size = tuples[1]
chatlearn.get_args().models['ppo_policy'].tensor_model_parallel_size = tuples[2]

chatlearn.get_args().runtime_args.colocation = [["policy", "ppo_policy"]]

policy = PolicyModel("policy")
ppo_policy = PPOPolicy("ppo_policy")

engine = CustomEngine(policy, ppo_policy)
engine.setup()
param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

assert param_sync_group.num_mapping == tuples[0] // tuples[2]

comm_pair_stage_1 = []
actor2rank = param_sync_group.actor2rank

for src, dst in param_sync_group.send_recv_actor_mappings.items():
    comm_pair_stage_1.append((actor2rank[src], actor2rank[dst[0]]))

assert comm_pair_stage_1 == [(0, 8), (1, 10), (2, 12), (3, 14), (4, 9), (5, 11), (6, 13), (7, 15)]

comm_pair_stage_2 = []
for src, dst in param_sync_group.send_recv_actor_mappings_stage2.items():
    comm_pair_stage_2.append((actor2rank[src], actor2rank[dst[0]]))

assert comm_pair_stage_2 == [(8, 9), (9, 8), (10, 11), (11, 10), (12, 13), (13, 12), (14, 15), (15, 14)]

print(f"pass test_case (dst_tp, src_pp, src_tp): {tuples}")

engine.model_manager.sync_parameters(requires_grad=False)
