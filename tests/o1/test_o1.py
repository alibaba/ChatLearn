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
"""entry file for training O1"""

import random
import os


import chatlearn
from chatlearn import ControlDependencies
from chatlearn import Evaluator, Trainer
from chatlearn.runtime.environment import MCTSEnv
from chatlearn.runtime.engine import Engine

import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import chatlearn
from chatlearn import TorchModule, BaseModule
from chatlearn.utils import future


class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.collate_fn = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"query": self.data[idx]}

class MCTS(BaseModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []

    def build_dataset(self, train_prompts, is_eval=False):
        dataset = CustomDataset(train_prompts)
        return dataset

    def init_tree(self):
        self.iter = 0

    def get_select_value_input(self, data):
        self._logger.info(f"[{self.replica_id}] perform get_select_value_input {self.iter}")
        data['query'] = [item + f"_step0_iter{self.iter}" for item in data['query']]
        return data
    
    def update(self, data):
        self._logger.info(f"[{self.replica_id}] perform update {self.iter}")
        data['query'] = [item + f"_step1_iter{self.iter}" for item in data['query']]
        return data
    
    def select_node_to_playout(self, data):
        self._logger.info(f"[{self.replica_id}] perform select_node_to_playout {self.iter}")
        data['query'] = [item + f"_step2_iter{self.iter}" for item in data['query']]
        return data

    def should_stop(self):
        return False

    def backpropagation(self, data):
        self._logger.info(f"[{self.replica_id}] perform backpropagation {self.iter}")
        data['query'] = [item + f"_step3_iter{self.iter}" for item in data['query']]
        self.iter += 1
        self.data.append(data)
        return data

    def get_data(self):
        return self.data
    
class PolicyModel(TorchModule):

    def forward_step(self, data, iteration):
        return data

    def playout(self, data):
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset

class RewardInference(TorchModule):

    def forward_step(self, data, iteration):
        return data

class RewardInference2(TorchModule):

    def forward_step(self, data, iteration):
        return data

class MCTSEngine(Engine):

    def __init__(self, mcts, policy, reward, reward2):

        def mcts_flow(batch):
            # selection
            select_out = mcts.get_select_value_input(batch)
            reward_out = reward.forward_step(select_out)
            # expansion
            mcts_out = mcts.update(reward_out)
            policy_out = policy.forward_step(mcts_out)
            mcts_expand_reward = reward.forward_step(policy_out)
            # playout
            to_playout = mcts.select_node_to_playout(mcts_expand_reward)
            policy_playout = policy.playout(to_playout)
            playout_reward = reward2.forward_step(policy_playout)
            # backpropagation
            mcts.backpropagation(policy_playout, playout_reward)
            return policy_playout

        env = MCTSEnv(mcts_flow, mcts)
        super().__init__(env, name='MCTS')


def test_o1():
    args = chatlearn.get_args()
    args.runtime_args.num_episode = 2
    args.runtime_args.max_iteration_per_sample = 2
    mcts_model = MCTS("mcts")
    policy_model = PolicyModel("policy")
    reward_model = RewardInference("reward")
    reward_model1 = RewardInference2("reward1")
    engine = MCTSEngine(mcts_model, policy_model, reward_model, reward_model1)
    all_prompts = [f'test{i}' for i in range(200)]
    split_ratio = 0.9 if args.runtime_args.eval_episode_interval > 0 else 1
    num_train = int(len(all_prompts) * split_ratio)
    train_prompts = all_prompts[:num_train]
    engine.set_dataset(train_prompts)
    engine.learn()
    all_data = []
    for replica in engine.named_models['mcts'].replicas:
        mcts_actors = replica.all_actors
        assert len(mcts_actors) == 1
        data = future.get(mcts_actors[0].get_data.remote())
        all_data.append(data)
    assert len(all_data) == args.runtime_args.sample_per_episode
    expected_results = [[] for i in range(args.runtime_args.sample_per_episode)]
    data_index = 0
    for episode_id in range(args.runtime_args.num_episode):
        for i in range(args.runtime_args.sample_per_episode):
            for j in range(args.runtime_args.max_iteration_per_sample):
                expected_results[i].append({'query': [f"test{data_index}_step0_iter{j}_step1_iter{j}_step2_iter{j}_step3_iter{j}"]})
            data_index += 1
    assert all_data == expected_results, f"all_data: {all_data} expected_results {expected_results}"
    engine.stop()

TEST_CASE = [test_o1]