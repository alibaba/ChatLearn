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
"""entry file for training Self Play RL"""

import chatlearn
from chatlearn.runtime.environment import SPRLEnv
from chatlearn.runtime.engine import Engine

import torch
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

class SPRL(BaseModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.separator = "|"
        self.iter = 0
        self.data = {}
        self.trajectory = {}
        self.trajectory['actor'] = []
        self.trajectory['critic'] = []

    def build_dataset(self, train_prompts, is_eval=False):
        dataset = CustomDataset(train_prompts)
        return dataset

    def reset(self):
        self.data = {}
        self.iter = 0

    def clear(self):
        self.iter = 0
        self.data = {}
        self.trajectory = {}
        self.trajectory['actor'] = []
        self.trajectory['critic'] = []

    def get_input(self, data):
        if len(self.data) == 0:
            return data
        return self.data

    def get_data(self):
        return self.trajectory

    def update(self, actor_out, critic_out, prm_out1, value_out1, prm_out2, value_out2):
        self._logger.info(f"[{self.replica_id}] perform update {self.iter}")

        trajectory_actor = self.concat_lists([actor_out['query'], prm_out1, value_out1])
        trajectory_critic = self.concat_lists([critic_out['query'], prm_out2, value_out2])
        self.trajectory['actor'].append(trajectory_actor)
        self.trajectory['critic'].append(trajectory_critic)
        self.data = critic_out
        self.iter += 1
        return critic_out

    def should_stop(self):
        return False

    def concat_lists(self, data):
        if not all(len(data[0]) == len(lst) for lst in data):
           raise ValueError(f"The element in data must be the same length, data: {data}")
        result = [self.separator.join(map(str, elements)) for elements in zip(*data)]
        return result

class ActorModel(TorchModule):

    def forward_step(self, data, iteration):
        data['query'] = [item + f"|actor" for item in data['query']]
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset

class ProcessRewardModel(TorchModule):

    def forward_step(self, data, iteration):
        prm = [f"prm_score" for item in data['query']]
        return prm

class CriticModel(TorchModule):

    def forward_step(self, data, iteration):
        data['query'] = [item + f"|critic" for item in data['query']]
        return data

class ValueModel(TorchModule):

    def forward_step(self, data, iteration):
        value = [f"value_score" for item in data['query']]
        return value

class SPRLEngine(Engine):

    def __init__(self, sprl, actor, critic, prm, value):

        def sprl_flow(batch):
            query = sprl.get_input(batch)
            # exec one iter
            actor_out = actor.forward_step(query)
            prm_out1 = prm.forward_step(actor_out)
            value_out1 = value.forward_step(actor_out)
            critic_out = critic.forward_step(actor_out)
            prm_out2 = prm.forward_step(critic_out)
            value_out2 = value.forward_step(critic_out)
            result = sprl.update(actor_out, critic_out, prm_out1, value_out1, prm_out2, value_out2)
            return result

        env = SPRLEnv(sprl_flow, sprl)
        super().__init__(env, name='SPRL')


if __name__ == "__main__":
    chatlearn.init()
    args = chatlearn.get_args()
    args.runtime_args.num_episode = 2
    args.runtime_args.max_iteration_per_sample = 2
    sprl_model = SPRL("sprl")
    actor_model = ActorModel("actor")
    critic_model = CriticModel("critic")
    prm_model = ProcessRewardModel("prm")
    value_model = ValueModel("value")

    engine = SPRLEngine(sprl_model, actor_model, critic_model, prm_model, value_model)
    all_prompts = [f'test{i}' for i in range(100)]
    split_ratio = 0.9 if args.runtime_args.eval_episode_interval > 0 else 1
    num_train = int(len(all_prompts) * split_ratio)
    train_prompts = all_prompts[:num_train]
    engine.set_dataset(train_prompts)
    engine.learn()
    all_data = []
    for replica in engine.named_models['sprl'].replicas:
        sprl_actors = replica.all_actors
        assert len(sprl_actors) == 1
        data = future.get(sprl_actors[0].get_data.remote())
        all_data.append(data)

    assert len(all_data[0]['actor']) == args.runtime_args.num_episode * args.runtime_args.sample_per_episode * args.runtime_args.max_iteration_per_sample
    assert len(all_data[0]['critic']) == args.runtime_args.num_episode * args.runtime_args.sample_per_episode * args.runtime_args.max_iteration_per_sample

    expected_results_actor = [[] for i in range(args.runtime_args.sample_per_episode)]
    expected_results_critic = [[] for i in range(args.runtime_args.sample_per_episode)]
    expected_results = {}
    expected_results['actor'] = []
    expected_results['critic'] = []
    data_index = 0
    for episode_id in range(args.runtime_args.num_episode):
        for i in range(args.runtime_args.sample_per_episode):
            expect_actor_str = f'test{data_index}|actor'
            expect_critic_str = f'test{data_index}|actor|critic'
            for j in range(args.runtime_args.max_iteration_per_sample):
                expected_results['actor'].append([expect_actor_str + "|prm_score|value_score"])
                expected_results['critic'].append([expect_critic_str + "|prm_score|value_score"])
                expect_actor_str += '|critic|actor'
                expect_critic_str += '|actor|critic'
            data_index += 1
    print("====actor trajectory:", all_data[0]['actor'], flush=True)
    print("====critic trajectory:", all_data[0]['critic'], flush=True)
    assert all_data[0] == expected_results
