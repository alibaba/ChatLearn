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
from chatlearn.runtime.engine import BaseEngine, Engine

import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import chatlearn
from chatlearn import RLHFEngine
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
        self.iter = 0

    def build_dataset(self, train_prompts, is_eval=False):
        dataset = CustomDataset(train_prompts)
        return dataset

    def backpropagation(self):
        self._logger.info(f"perform backpropagation {self.iter}")
        self.iter += 1
        return {}

    def get_select_value_input(self, data):
        return data
    
    def update(self, data):
        return data
    
    def select_node_to_playout(self, data):
        return data

    def should_stop(self):
        return False

    
class PolicyModel(TorchModule):

    def forward_step(self, data, iteration):
        return data

    def playout(self, data):
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset


class PolicyReference(TorchModule):

    def forward_step(self, data, iteration):
        return data


class RewardInference(TorchModule):

    def forward_step(self, data, iteration):
        return data


class ValueInference(TorchModule):

    def forward_step(self, data, iteration):
        return data


class PolicyTrainer(TorchModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []

    def _get_rank(self):
        return int(os.environ["RANK"])

    @property
    def data_parallel_size(self):
        return 2

    @property
    def data_parallel_rank(self):
        if self._get_rank() < 4:
            return 0
        return 1

    def train_step(self, data, iteration):
        print(f"ppo policy train_step ========= {self.data_parallel_rank}", flush=True)
        if self._get_rank() == 0 or self._get_rank() == 4:
            self.data.append(data)
        num_mb = len(data)
        return num_mb

    def get_data(self):
        return self.data


class ValueTrainer(TorchModule):

    @property
    def data_parallel_size(self):
        return 2

    @property
    def data_parallel_rank(self):
        if int(os.environ["RANK"]) < 4:
            return 0
        return 1

    def train_step(self, data, iteration):
        print("ppo value train_step =========", flush=True)
        num_mb = len(data)
        return num_mb

class MCTSEngine(Engine):

    def __init__(self, mcts, policy, reference, reward, value):

        def mcts_flow(batch):
            # selection
            select_out = mcts.get_select_value_input(batch)
            value_out = value.forward_step(select_out)
            # expansion
            mcts_out = mcts.update(value_out)
            string = policy.forward_step(mcts_out)
            ref_out = reference.forward_step(string)
            mcts_expand_reward = reward.forward_step(ref_out)
            # playout
            to_playout = mcts.select_node_to_playout(mcts_expand_reward)
            policy_playout = policy.playout(to_playout)
            playout_reward = reward.forward_step(policy_playout)
            # backpropagation
            with ControlDependencies(playout_reward):
                mcts.backpropagation()
            return policy_playout, playout_reward

        env = MCTSEnv(mcts_flow, mcts)
        super().__init__(env, name='MCTS')


if __name__ == "__main__":
    chatlearn.init()
    args = chatlearn.get_args()
    args.runtime_args.num_episode = 2
    args.runtime_args.max_iteration_per_batch = 2
    args.runtime_args.debug = True
    mcts_model = MCTS("mcts")
    reference_model = PolicyReference("reference")
    policy_model = PolicyModel("policy")
    reward_model = RewardInference("reward")

    value_model = ValueInference("value")
    engine = MCTSEngine(mcts_model, policy_model, reference_model, reward_model, value_model)
    all_prompts = ['test'] * 200
    split_ratio = 0.9 if args.runtime_args.eval_episode_interval > 0 else 1
    num_train = int(len(all_prompts) * split_ratio)
    random.shuffle(all_prompts)
    train_prompts = all_prompts[:num_train]
    engine.set_dataset(train_prompts)
    engine.learn()
