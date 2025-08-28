# Copyright 2024 Alibaba-inc. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
"""rule reward"""
from typing import Dict, List

import torch

from chatlearn import BaseModule
from chatlearn.utils.rule_reward_score import math
from chatlearn.runtime.decorator import timeit, compute_decorator

class RuleReward(BaseModule):
    """rule reward"""
    # pylint: disable=abstract-method

    def __init__(self, name: str, args=None, replica_id: int=0):
        """The chatlearn wrapper for a RuleReward model.

        Args:
            name (str): The name of this module
            args (Any, optional): The arguments. Defaults to None.
            replica_id (int, optional): The replica id of this module. Defaults to 0.
        """
        super().__init__(name, args=args, replica_id=replica_id)
        assert self.total_gpu == 0, "RuleReward does not require GPU"
        self._num_gpu_per_replica = 0
        self._num_replica = self.module_args.num_cpu // self.module_args.cpu_per_process

    def setup(self):
        self.stats = {}
        self._metric_prefix = "rule_reward"

    def _forward_step(self, data: List) -> torch.Tensor:
        # str_prompts_list = data["str_prompts"]
        self._logger.info(f"RuleReward _forward_step Num of request: {len(data)}")

        reward = []

        for data_b in data:
            str_output = data_b["str_outputs"]
            data_source = data_b["data_source"]
            ground_truth = data_b["ground_truth"]
            compute_score_fn = self.select_rule_reward_score_fn(data_source)
            reward.append(compute_score_fn(str_output, ground_truth))
            data_b.update({"rule_reward": reward[-1], "eval_source": data_source})
        return data, reward

    @compute_decorator(trainable=False, rollout=False)
    @timeit()
    def forward_step(self, data: Dict, iteration=0, **kwargs) -> Dict: # pylint: disable=unused-argument
        data, reward = self._forward_step(data)

        # collect stats
        train_reward_score = sum(reward) / len(reward)
        train_reward_stats = {
            "train_reward_score": train_reward_score,
        }
        self._metric_list.append(train_reward_stats)
        return data

    @compute_decorator(trainable=False, rollout=False)
    @timeit()
    def eval_forward(self, data: Dict, **kwargs) -> Dict: # pylint: disable=unused-argument

        return self._forward_step(data)[0]

    def select_rule_reward_score_fn(self, data_source: str):
        if data_source in ['openai/gsm8k', 'DigitalLearningGmbH/MATH-lighteval', 'aime24', 'aime25']:
            return math.compute_score
        else:
            raise NotImplementedError
