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
Test case: (dst_tp, dst_pp, src_tp, src_pp) = (8, 1, 2, 4), and validate results of sync params."""

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

    def expert_parallel_rank(self):
        return 0

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

def _create_engine(parallel_size : tuple):
    for _, model_config in chatlearn.get_args().models.items():
        model_config.num_gpu = 8
    chatlearn.get_args().models['policy'].tensor_model_parallel_size = parallel_size[0]
    chatlearn.get_args().models['policy'].pipeline_model_parallel_size = parallel_size[1]
    chatlearn.get_args().models['ppo_policy'].tensor_model_parallel_size = parallel_size[2]
    chatlearn.get_args().models['ppo_policy'].pipeline_model_parallel_size = parallel_size[3]

    chatlearn.get_args().runtime_args.colocation = [["policy", "ppo_policy"]]

    policy = PolicyModel("policy")
    ppo_policy = PPOPolicy("ppo_policy")

    engine = CustomEngine(policy, ppo_policy)
    engine.set_dataset(["Mock dataset"])
    engine.setup()
    return engine


def test_same_tp_same_pp():
     # parallel_size: (dst_tp, dst_pp, src_tp, src_pp)
    parallel_size = (8, 1, 8, 1)
    engine = _create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.tp_num_mapping == parallel_size[0] // parallel_size[2]

    comm_pair_stage_1 = []
    actor2rank = param_sync_group.actor2rank

    for src, dsts in param_sync_group.send_recv_actor_mappings.items():
        for dst in dsts:
            comm_pair_stage_1.append((actor2rank[src], actor2rank[dst]))

    print(f"comm_pair_stage_1: {comm_pair_stage_1}")

    assert comm_pair_stage_1 == [(0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)]

    comm_pair_stage_2 = []
    for src, dsts in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst in dsts:
            comm_pair_stage_2.append((actor2rank[src], actor2rank[dst]))

    print(f"comm_pair_stage_2: {comm_pair_stage_2}")
    assert comm_pair_stage_2 == []
    engine.stop()
    print(f"pass test_case (dst_tp, src_pp, src_tp): {parallel_size}")

def test_unbalanced_tp():
    # parallel_size: (dst_tp, dst_pp, src_tp, src_pp)
    parallel_size = (8, 1, 4, 2)

    engine = _create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.tp_num_mapping == parallel_size[0] // parallel_size[2]

    comm_pair_stage_1 = []
    actor2rank = param_sync_group.actor2rank

    for src, dst in param_sync_group.send_recv_actor_mappings.items():
        comm_pair_stage_1.append((actor2rank[src], actor2rank[dst[0]]))

    comm_pair_stage_1.sort(key=lambda x: x[0])
    assert comm_pair_stage_1 == [(0, 8), (1, 10), (2, 12), (3, 14), (4, 9), (5, 11), (6, 13), (7, 15)], \
        f"Real Value {comm_pair_stage_1}"

    comm_pair_stage_2 = []
    for src, dst in param_sync_group.send_recv_actor_mappings_stage2.items():
        comm_pair_stage_2.append((actor2rank[src], actor2rank[dst[0]]))

    comm_pair_stage_2.sort(key=lambda x: x[0])
    assert comm_pair_stage_2 == [(8, 9), (9, 8), (10, 11), (11, 10), (12, 13), (13, 12), (14, 15), (15, 14)], \
        f"Real Value {comm_pair_stage_2}"

    print(f"pass test_case (dst_tp, src_pp, src_tp): {parallel_size}")
    engine.stop()

def test_unbalanced_tp_1():
    # parallel_size: (dst_tp, dst_pp, src_tp, src_pp)
    parallel_size = (8, 1, 2, 4)

    engine = _create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.tp_num_mapping == parallel_size[0] // parallel_size[2]

    comm_pair_stage_1 = []
    actor2rank = param_sync_group.actor2rank

    for src, dsts in param_sync_group.send_recv_actor_mappings.items():
        for dst in dsts:
            comm_pair_stage_1.append((actor2rank[src], actor2rank[dst]))

    # TODO The original test case, is this a bug?
    # [(0, 8), (1, 12), (2, 9), (3, 13), (4, 10), (5, 14), (6, 11), (7, 15)], \
    assert comm_pair_stage_1 == \
        [(1, 12), (3, 13), (4, 8), (6, 10), (0, 11), (2, 9), (5, 15), (7, 14)], \
        f"Real Value {comm_pair_stage_1}"

    comm_pair_stage_2 = []
    for src, dsts in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst in dsts:
            comm_pair_stage_2.append((actor2rank[src], actor2rank[dst]))

    assert comm_pair_stage_2 == \
        [(8, 9), (8, 10), (8, 11), (9, 8), (9, 10), (9, 11), \
        (10, 8), (10, 9), (10, 11), (11, 8), (11, 9), (11, 10), \
        (12, 13), (12, 14), (12, 15), (13, 12), (13, 14), (13, 15), \
        (14, 12), (14, 13), (14, 15), (15, 12), (15, 13), (15, 14)], \
        f"Real Value {comm_pair_stage_2}"

    print(f"pass test_case (dst_tp, src_pp, src_tp): {parallel_size}")
    engine.stop()

def test_unbalanced_tp_2():
    # parallel_size: (dst_tp, dst_pp, src_tp, src_pp)
    parallel_size = (8, 1, 4, 1)
    engine = _create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.tp_num_mapping == parallel_size[0] // parallel_size[2]

    comm_pair_stage_1 = []
    actor2rank = param_sync_group.actor2rank

    for src, dsts in param_sync_group.send_recv_actor_mappings.items():
        for dst in dsts:
            comm_pair_stage_1.append((actor2rank[src], actor2rank[dst]))

    comm_pair_stage_1.sort(key=lambda x: x[0])
    assert comm_pair_stage_1 == \
        [(0, 8), (1, 10), (2, 12), (3, 14)], f"Real Value {comm_pair_stage_1}"


    comm_pair_stage_2 = []
    for src, dsts in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst in dsts:
            comm_pair_stage_2.append((actor2rank[src], actor2rank[dst]))

    assert comm_pair_stage_2 == \
        [(8, 9), (10, 11), (12, 13), (14, 15)], f"Real Value {comm_pair_stage_2}"


    print(f"pass test_case (dst_tp, src_pp, src_tp): {parallel_size}")
    engine.stop()

def test_unbalanced_tp_3():
    # parallel_size: (dst_tp, dst_pp, src_tp, src_pp)
    parallel_size = (8, 1, 2, 1)

    engine = _create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.tp_num_mapping == parallel_size[0] // parallel_size[2]

    comm_pair_stage_1 = []
    actor2rank = param_sync_group.actor2rank

    for src, dsts in param_sync_group.send_recv_actor_mappings.items():
        for dst in dsts:
            comm_pair_stage_1.append((actor2rank[src], actor2rank[dst]))

    assert comm_pair_stage_1 == [(0, 8), (1, 12)], f"Real Value {comm_pair_stage_1}"


    comm_pair_stage_2 = []
    for src, dsts in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst in dsts:
            comm_pair_stage_2.append((actor2rank[src], actor2rank[dst]))

    assert comm_pair_stage_2 == \
        [(8, 9), (8, 10), (8, 11), (12, 13), (12, 14), (12, 15)], f"Real Value {comm_pair_stage_2}"


    print(f"pass test_case (dst_tp, src_pp, src_tp): {parallel_size}")
    engine.stop()

def test_unbalanced_tp_4():
    # parallel_size: (dst_tp, dst_pp, src_tp, src_pp)
    parallel_size = (4, 2, 2, 4)

    engine = _create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.tp_num_mapping == parallel_size[0] // parallel_size[2]

    comm_pair_stage_1 = []
    actor2rank = param_sync_group.actor2rank

    for src, dsts in param_sync_group.send_recv_actor_mappings.items():
        for dst in dsts:
            comm_pair_stage_1.append((actor2rank[src], actor2rank[dst]))

    print(f"comm_pair_stage_1: {comm_pair_stage_1}")

    assert comm_pair_stage_1 == [(0, 8), (1, 10), (2, 9), (3, 11), (4, 13), (5, 15), (6, 12), (7, 14)], \
        f"Real Value {comm_pair_stage_1}"


    comm_pair_stage_2 = []
    for src, dsts in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst in dsts:
            comm_pair_stage_2.append((actor2rank[src], actor2rank[dst]))

    assert comm_pair_stage_2 == \
        [(8, 9), (9, 8), (10, 11), (11, 10), (12, 13), (13, 12), (14, 15), (15, 14)], \
        f"Real Value {comm_pair_stage_2}"

    engine.stop()
    print(f"pass test_case (dst_tp, src_pp, src_tp): {parallel_size}")

def test_unbalanced_tp_5():
    # parallel_size: (dst_tp, dst_pp, src_tp, src_pp)
    parallel_size = (4, 1, 1, 2)

    engine = _create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.tp_num_mapping == parallel_size[0] // parallel_size[2]

    comm_pair_stage_1 = []
    actor2rank = param_sync_group.actor2rank

    for src, dsts in param_sync_group.send_recv_actor_mappings.items():
        for dst in dsts:
            comm_pair_stage_1.append((actor2rank[src], actor2rank[dst]))

    print(f"comm_pair_stage_1: {comm_pair_stage_1}")

    assert comm_pair_stage_1 == [(0, 8), (1, 9), (2, 12), (3, 13)], \
        f"Real Value {comm_pair_stage_1}"

    comm_pair_stage_2 = []
    for src, dsts in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst in dsts:
            comm_pair_stage_2.append((actor2rank[src], actor2rank[dst]))

    print(f"comm_pair_stage_2: {comm_pair_stage_2}")

    assert comm_pair_stage_2 == \
        [(8, 9), (8, 10), (8, 11), (9, 8), (9, 10), (9, 11), (12, 13), (12, 14), (12, 15), (13, 12), (13, 14), (13, 15)], \
        f"Real: {comm_pair_stage_2}"

    print(f"pass test_case (dst_tp, src_pp, src_tp): {parallel_size}")
    engine.stop()

def test_unbalanced_tp_6():
    # parallel_size: (dst_tp, dst_pp, src_tp, src_pp)
    parallel_size = (2, 1, 1, 1)

    engine = _create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.tp_num_mapping == parallel_size[0] // parallel_size[2]

    comm_pair_stage_1 = []
    actor2rank = param_sync_group.actor2rank

    for src, dsts in param_sync_group.send_recv_actor_mappings.items():
        for dst in dsts:
            comm_pair_stage_1.append((actor2rank[src], actor2rank[dst]))

    print(f"comm_pair_stage_1: {comm_pair_stage_1}")

    assert comm_pair_stage_1 == [(0, 8), (1, 10), (2, 12), (3, 14)], \
        f"Real Value {comm_pair_stage_1}"


    comm_pair_stage_2 = []
    for src, dsts in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst in dsts:
            comm_pair_stage_2.append((actor2rank[src], actor2rank[dst]))

    print(f"comm_pair_stage_2: {comm_pair_stage_2}")

    assert comm_pair_stage_2 == \
        [(8, 9), (10, 11), (12, 13), (14, 15)], f"Real Value {comm_pair_stage_2}"


    print(f"pass test_case (dst_tp, src_pp, src_tp): {parallel_size}")
    engine.stop()


TEST_CASE = [
    test_same_tp_same_pp,
    test_unbalanced_tp,
    test_unbalanced_tp_1,
    test_unbalanced_tp_2,
    test_unbalanced_tp_3,
    test_unbalanced_tp_4,
    test_unbalanced_tp_5,
    test_unbalanced_tp_6
    ]

if __name__ == "__main__":
    chatlearn.init()
    for case in TEST_CASE:
        case()
