# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
"""Trainer"""

import math
import ray

from chatlearn.utils import future
from chatlearn.utils.logger import logger


class BaseTrainer:
    """
    base trainer
    """

    def __init__(self, args):
        self.args = args


class PPOTrainer(BaseTrainer):
    """
    PPO Trainer
    """

    def __init__(self, args, ppo_policy_model, ppo_value_model):
        super().__init__(args)
        self.ppo_policy_model = ppo_policy_model
        self.ppo_value_model = ppo_value_model
        self.models = [ppo_policy_model, ppo_value_model]
        self.args = args
        self.num_micro_batch = args.train_global_batch_size // args.train_micro_batch_size
        self.iteration = 0
        self.ppo_policy_name = ppo_policy_model.name
        self.ppo_value_name = ppo_value_model.name
        self.names2colocate = {
            self.ppo_policy_name: False,
            self.ppo_value_name: False
        }
        self.is_ppo_models_colocate = True

    def setup(self, model_packs=None):
        for group in self.args.colocation:
            for model in group:
                if model in self.names2colocate:
                    self.names2colocate[model] = (len(group) > 1)
        for model_pack in model_packs:
            model_name_pack = [model.name for model in model_pack]
            if len(model_name_pack) > 1 \
                    and self.ppo_policy_name in model_name_pack \
                    and self.ppo_value_name in model_name_pack:
                self.is_ppo_models_colocate = False

    def train_step(self, train_data, train_info, wait=True, to_empty_value_cache=False, to_empty_policy_cache=False):
        ref0 = self.ppo_value_model.train_step(train_data, train_info, to_empty_cache=to_empty_value_cache)
        ref1 = self.ppo_policy_model.train_step(train_data, train_info, to_empty_cache=to_empty_policy_cache)
        if wait:
            future.wait(ref0 + ref1)
        else:
            return [ref0[-1], ref1[-1]]

    def set_data_loader(self, data_loader):
        self._data_loader = data_loader

    def next_batch(self):
        batches = []
        for _ in range(self.num_micro_batch):
            data = self._data_loader.next.remote()
            if future.get(self._data_loader.has_next.remote()):
                batches.append(data)
        if not batches:
            return
        else:
            if len(batches) < self.num_micro_batch:
                batches += batches[:self.num_micro_batch - len(batches)]
            return batches

    def num_training_iteration(self):
        # Given that we have incorporated support for relay buffer and dynamic reward outputs,
        # the number of training data batches per episode may differ, hence we dynamically determine the total number of batches per episode.
        _sample_per_episode = ray.get(self._data_loader.total_samples.remote())
        return math.ceil(_sample_per_episode / self.args.train_global_batch_size)

    def train(self, episode):
        _num_training_iteration = self.num_training_iteration()
        for epoch in range(self.args.num_training_epoch):
            if epoch > 0:
                ret = self._data_loader.shuffle.remote()
                future.wait(ret)
            if not self.is_ppo_models_colocate:
                logger.info(f"{self.ppo_policy_model.name} and {self.ppo_value_model.name} execute concurrently")
                train_refs = []
                train_datas = [self.next_batch() for step in range(_num_training_iteration)]
                train_data_len = len(train_datas)
                is_last_epoch = (epoch == self.args.num_training_epoch - 1)
                for index, train_data in enumerate(train_datas):
                    if train_data:
                        train_info = {"iteration": self.iteration}
                        is_last_index = (index == train_data_len - 1)
                        is_last_epoch_last_index = (is_last_epoch and is_last_index)
                        refs = self.train_step(
                            train_data, train_info, wait=False,
                            to_empty_value_cache=self.names2colocate[self.ppo_value_name] and is_last_epoch_last_index,
                            to_empty_policy_cache=self.names2colocate[self.ppo_policy_name] and is_last_epoch_last_index
                        )
                        train_refs.extend(refs)
                        self.iteration += 1
                future.wait(train_refs, 'ppo training')
            else:
                logger.info(f"{self.ppo_policy_model.name} and {self.ppo_value_model.name} execute serially")
                batches = []
                for step in range(_num_training_iteration):
                    train_data = self.next_batch()
                    if train_data:
                        batches.append(train_data)
                cur_iteration = self.iteration
                results = []
                batch_len = len(batches)
                for index, batch in enumerate(batches):
                    train_info = {"iteration": cur_iteration}
                    to_empty_cache = (index == batch_len - 1)
                    value_loss = self.ppo_value_model.train_step(batch, train_info, to_empty_cache=to_empty_cache)
                    results.append(value_loss[-1])
                    cur_iteration += 1
                future.wait(results, desc=" ".join(model.name for model in [self.ppo_value_model]))
                cur_iteration = self.iteration
                results = []
                for index, batch in enumerate(batches):
                    train_info = {"iteration": cur_iteration}
                    to_empty_cache = (index == batch_len - 1)
                    policy_loss = self.ppo_policy_model.train_step(batch, train_info, to_empty_cache=to_empty_cache)
                    results.append(policy_loss[-1])
                    cur_iteration += 1
                future.wait(results, desc=" ".join(model.name for model in [self.ppo_policy_model]))
                self.iteration = cur_iteration
                logger.info(f"train episode: {episode}, epoch {epoch} step {step} iteration {self.iteration}")
