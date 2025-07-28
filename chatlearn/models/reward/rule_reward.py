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
from typing import Dict

import torch

from chatlearn import BaseModule
from chatlearn.utils.rule_reward_score import math

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
        self._metric_prefix = "rulereward"

    def _forward_step(self, data: Dict) -> torch.Tensor:

        # str_prompts_list = data["str_prompts"]
        str_outputs_list = data["str_outputs"]
        data_source_list = data["data_source"]
        ground_truth_list = data["ground_truth"]
        self._logger.info(f"RuleReward _forward_step Num of request: {len(str_outputs_list)}")

        reward_tensor = torch.zeros([len(str_outputs_list), 1], dtype=torch.float32)
        eval_source = []

        for i, str_output in enumerate(str_outputs_list):
            data_source = data_source_list[i]
            ground_truth = ground_truth_list[i]
            compute_score_fn = self.select_rule_reward_score_fn(data_source)
            reward_tensor[i] = compute_score_fn(str_output, ground_truth)
            eval_source.append(data_source)
        data["rule_rewards"] = reward_tensor
        data["eval_source"] = eval_source
        return data

    def forward_step(self, data: Dict, iteration=0) -> Dict:

        res_dict = self._forward_step(data)

        # collect stats
        rule_rewards = res_dict["rule_rewards"]
        train_reward_score = rule_rewards.mean().item()
        train_reward_stats = {
            "train_reward_score": train_reward_score,
        }
        self._metric_list.append(train_reward_stats)
        return res_dict

    def eval_forward(self, data: Dict) -> Dict:

        return self._forward_step(data)

    def select_rule_reward_score_fn(self, data_source: str):
        if data_source in ['openai/gsm8k', 'DigitalLearningGmbH/MATH-lighteval', 'aime24', 'aime25']:
            return math.compute_score
        else:
            raise NotImplementedError
