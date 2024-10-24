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
"""Test when trainer_tp < inference_tp but trainer_tp can divide inference_tp.
Test case: (dst_tp, dst_pp, src_tp, src_pp) = (8, 1, 2, 1), and validate results of sync params."""

import os
import ray
import time

import torch
from collections import defaultdict

import chatlearn
from chatlearn import TorchModule
from chatlearn.runtime.engine import Engine
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.trainer import Trainer
from chatlearn.utils import future


class TestTorchModule(TorchModule):
    def get_local_param_ranks(self):
        return [self._get_rank()], 0

    def tensor_parallel_rank(self):
        return int(os.environ["RANK"])

    def pipeline_parallel_rank(self):
        return self.tensor_parallel_rank() // self.tensor_model_parallel_size()


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

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        query = data["query"]
        bs = query.size(0)
        data["policy_out"] = torch.ones([bs, 1024]).cuda()
        return data


class PPOPolicy(TestTorchModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []

    def _get_rank(self):
        return int(os.environ["RANK"])

    @property
    def data_parallel_size(self):
        return 8 // (self.tensor_model_parallel_size() * self.pipeline_model_parallel_size())


    @property
    def data_parallel_rank(self):
        return self._get_rank() // self.tensor_model_parallel_size()


# tuples: (dst_tp, dst_pp, src_tp, src_pp)
tuples = (8, 1, 2, 1)

chatlearn.init()
for _, model_config in chatlearn.get_args().models.items():
    model_config.num_gpu = 8
chatlearn.get_args().models['policy'].tensor_model_parallel_size = tuples[0]
chatlearn.get_args().models['policy'].pipeline_model_parallel_size = tuples[1]
chatlearn.get_args().models['ppo_policy'].tensor_model_parallel_size = tuples[2]
chatlearn.get_args().models['ppo_policy'].pipeline_model_parallel_size = tuples[3]


chatlearn.get_args().runtime_args.colocation = [["policy", "ppo_policy"]]

policy = PolicyModel("policy")
ppo_policy = PPOPolicy("ppo_policy")

engine = CustomEngine(policy, ppo_policy)
engine.setup()
param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

assert param_sync_group.tp_num_mapping == tuples[0] // tuples[2]

comm_pair_stage_1 = []
actor2rank = param_sync_group.actor2rank

for src, dsts in param_sync_group.send_recv_actor_mappings.items():
    for dst in dsts:
        comm_pair_stage_1.append((actor2rank[src], actor2rank[dst]))

assert comm_pair_stage_1 == [(0, 8), (1, 12)]

comm_pair_stage_2 = []
for src, dsts in param_sync_group.send_recv_actor_mappings_stage2.items():
    for dst in dsts:
        comm_pair_stage_2.append((actor2rank[src], actor2rank[dst]))

assert comm_pair_stage_2 == \
    [(8, 9), (8, 10), (8, 11), (12, 13), (12, 14), (12, 15)]

print(f"pass test_case (dst_tp, src_pp, src_tp): {tuples}")
