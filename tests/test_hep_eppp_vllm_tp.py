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
"""
Test:
1. trainer_tp <= inference_tp
2. trainer_pp > inference_pp
2. trainer_ep > 1 while inference_ep = 1
3. HEP is enabled for trainer but disabled for inference.

Current test case: 
(dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp) = (1, 4, 1, 4, 1, 2).
"""

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
from chatlearn.schedule.resource_manager import ResourceManager
from chatlearn.schedule.model_manager import ModelManager
from chatlearn.synchronizer.parameter_sync import ParameterSyncGroupwithHEP


class TestTorchModule(TorchModule):

    def _get_rank(self):
        return int(os.environ["RANK"])

    def get_local_param_ranks(self):
        global_rank = self._get_rank()
        data_modulo_expert_parallel_ranks = [global_rank]
        return data_modulo_expert_parallel_ranks, 0

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

    def _create_remote_models(self):
        """
        :meta private:
        """
        resource_manager = ResourceManager(self._models)
        self.model_manager = MockModelManagerHEP(self._models, resource_manager, self.global_args)
        for src_model, dst_model in self._param_sync_pairs:
            self.model_manager.set_parameter_sync(src_model, dst_model)
        self.model_manager.remote()
        self.remote_models = self.model_manager.dist_models
        self.named_models = {model.name: model for model in self.remote_models}


class MockModelManagerHEP(ModelManager):

    def build_parameter_group(self):
        for src_model, dst_model in self._parameter_sync_model_pair:
            group_name = self._get_group_name(src_model, dst_model)
            sync_frequency = self._get_sync_frequency(dst_model)
            sync_group = MockParameterSyncGroupwithHEP(
                self._name2distmodel[src_model.name],
                self._name2distmodel[dst_model.name],
                group_name,
                sync_frequency,
                self.error_signal
            )
            self.parameter_sync_groups[group_name] = sync_group


class MockParameterSyncGroupwithHEP(ParameterSyncGroupwithHEP):

    def setup_rank_mapping(self):
        self.tp_num_mapping = self.num_dst_tensor_parallel // self.num_src_tensor_parallel
        self.ep_num_mapping = self.num_dst_expert_parallel / self.num_src_expert_parallel
        self.hep_num_mapping = self.num_dst_hyper_expert_parallel / self.num_src_hyper_expert_parallel

        self.build_rank_mapping_for_ep(add_recv_actor_fn=self.empty_add_recv_actor)
        self.build_rank_mapping_two_stage()


class PolicyModel(TestTorchModule):

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        query = data["query"]
        bs = query.size(0)
        data["policy_out"] = torch.ones([bs, 1024]).cuda()
        return data

    @property
    def data_parallel_size(self):
        return 2

    @property
    def data_parallel_rank(self):
        return int(self._get_rank() // 4)

    def tensor_parallel_rank(self):
        return int(self._get_rank() % 4)

    def expert_parallel_rank(self):
        return 0

    def pipeline_parallel_rank(self):
        return 0

    def tensor_and_expert_model_parallel_size(self):
        return 4


class PPOPolicy(TestTorchModule):

    @property
    def data_parallel_size(self):
        return 4

    @property
    def data_parallel_rank(self):
        return self._get_rank() % 4

    def tensor_parallel_rank(self):
        return 0

    def expert_parallel_rank(self):
        return self._get_rank() % 4

    def pipeline_parallel_rank(self):
        return self._get_rank() // 4

    def tensor_and_expert_model_parallel_size(self):
        return 4

def test_hep_ep_vllm_tp_dst_ep1_tp4_pp1_src_ep4_tp1_pp2():
    # tuples: (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp)
    tuples = (1, 4, 1,
              4, 1, 2)

    chatlearn.init()
    for _, model_config in chatlearn.get_args().models.items():
        model_config.num_gpu = 8
    chatlearn.get_args().models['policy'].expert_model_parallel_size = tuples[0]
    chatlearn.get_args().models['policy'].tensor_model_parallel_size = tuples[1]
    chatlearn.get_args().models['policy'].pipeline_model_parallel_size = tuples[2]
    chatlearn.get_args().models['ppo_policy'].expert_model_parallel_size = tuples[3]
    chatlearn.get_args().models['ppo_policy'].tensor_model_parallel_size = tuples[4]
    chatlearn.get_args().models['ppo_policy'].pipeline_model_parallel_size = tuples[5]

    chatlearn.get_args().runtime_args.colocation = [["policy", "ppo_policy"]]

    policy = PolicyModel("policy")
    ppo_policy = PPOPolicy("ppo_policy")

    os.environ['QWEN_VERSION'] = "qwen_moe_v1" # enable ParameterSyncGroupwithHEP

    engine = CustomEngine(policy, ppo_policy)
    engine.setup()
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    # Judge num mappings
    assert param_sync_group.ep_num_mapping == tuples[0] / tuples[3]
    assert param_sync_group.tp_num_mapping == tuples[1] // tuples[4]

    # Judge allgather actors
    allgather_actors = param_sync_group.send_actors_to_allgather_routed_experts
    actor2rank = param_sync_group.actor2rank

    assert len(allgather_actors) == 2
    assert len(allgather_actors[0]) == 4 # prev 4 src ranks should all-gather routed experts
    assert len(allgather_actors[1]) == 4 # last 4 src ranks should all-gather routed experts
    assert len(actor2rank) == 16 # all of the 16 actors should have rank
    assert len(set(list(actor2rank.values()))) == len(actor2rank) # all ranks should be unique

    allgather_actor_ranks = []
    for actor_list in allgather_actors:
        allgather_actor_ranks.append([])
        for actor in actor_list:
            allgather_actor_ranks[-1].append(actor2rank[actor])

    assert allgather_actor_ranks == [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Judge src->dst rank mappings
    comm_pairs_for_except_routed_experts_stage1 = []
    comm_pairs_for_except_routed_experts_stage2 = []
    
    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings.items():
        for dst_rank in dst_ranks:
            comm_pairs_for_except_routed_experts_stage1.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst_rank in dst_ranks:
            comm_pairs_for_except_routed_experts_stage2.append((actor2rank[src_rank], actor2rank[dst_rank]))

    assert comm_pairs_for_except_routed_experts_stage1 == [(0, 8), (4, 9), (1, 12), (5, 13)]
    assert comm_pairs_for_except_routed_experts_stage2 == [
        (8, 9), (8, 10), (8, 11),
        (9, 8), (9, 10), (9, 11),
        (12, 13), (12, 14), (12, 15),
        (13, 12), (13, 14), (13, 15)
    ]

    print(f"pass test_case (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp): {tuples}")

test_hep_ep_vllm_tp_dst_ep1_tp4_pp1_src_ep4_tp1_pp2()
