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

from rlhf.utils import future
from rlhf.utils.logger import logger


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
        self.num_training_iteration = math.ceil(args.sample_per_episode / args.train_global_batch_size)
        self.num_micro_batch = args.train_global_batch_size // args.train_micro_batch_size
        self.iteration = 0
        model_names = [m.name for m in self.models]
        self._colocation = False
        for group in self.args.colocation:
            new_group = []
            for model in group:
                if model in model_names:
                    new_group.append(model)
            if len(new_group) > 1:
                self._colocation = True

    def setup(self):
        pass

    def train_step(self, train_data, train_info, wait=True):
        ref0 = self.ppo_value_model.train_step(train_data, train_info)
        ref1 = self.ppo_policy_model.train_step(train_data, train_info)
        if wait:
            future.wait(ref0 + ref1)
        else:
            return [ref0[0], ref1[0]]

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

    def wait_and_empty_cache(self, models, results):
        desc = " ".join(model.name for model in models)
        future.wait(results, desc)
        # empty cache, so that other models can use
        refs = []
        for model in models:
            refs.extend(model.empty_cache())
        future.wait(refs)

    def train(self, episode):
        for epoch in range(self.args.num_training_epoch):
            if epoch > 0:
                ret = self._data_loader.shuffle.remote()
                future.wait(ret)
            if not self._colocation:
                logger.info(f"{self.ppo_policy_model.name} and {self.ppo_value_model.name} execute concurrently")
                train_refs = []
                train_datas = [self.next_batch() for step in range(self.num_training_iteration)]
                for train_data in train_datas:
                    if train_data:
                        train_info = {"iteration": self.iteration}
                        train_refs.extend(self.train_step(train_data, train_info, wait=False))
                        self.iteration += 1
                future.wait(train_refs, 'ppo training')
                if self.args.colocation and epoch == self.args.num_training_epoch - 1:
                    value_cache_refs = self.ppo_value_model.empty_cache()
                    policy_cache_refs = self.ppo_policy_model.empty_cache()
                    future.wait(value_cache_refs + policy_cache_refs)
            else:
                logger.info(f"{self.ppo_policy_model.name} and {self.ppo_value_model.name} execute serially")
                batches = []
                for step in range(self.num_training_iteration):
                    train_data = self.next_batch()
                    if train_data:
                        batches.append(train_data)
                cur_iteration = self.iteration
                results = []
                for batch in batches:
                    train_info = {"iteration": cur_iteration}
                    value_loss = self.ppo_value_model.train_step(batch, train_info)
                    results.append(value_loss[0])
                    cur_iteration += 1
                self.wait_and_empty_cache([self.ppo_value_model], results)
                cur_iteration = self.iteration
                results = []
                for batch in batches:
                    train_info = {"iteration": cur_iteration}
                    policy_loss = self.ppo_policy_model.train_step(batch, train_info)
                    results.append(policy_loss[0])
                    cur_iteration += 1
                self.wait_and_empty_cache([self.ppo_policy_model], results)
                self.iteration = cur_iteration
                logger.info(f"train episode: {episode}, epoch {epoch} step {step} iteration {self.iteration}")
