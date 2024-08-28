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
"""reward math model"""
from collections import defaultdict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from chatlearn import BaseModule
from .utils import tensorboard_scalar_dict
from .constants import RunningMoments, get_running_stats, reset_running_stats
from .rm_sys.math_rule_rm import MathRuleRM

class MathReward(BaseModule):
    """Math reward"""

    def setup(self):
        self.math_rule_rm = MathRuleRM()
        self.stats = {}
        self.running = RunningMoments()
        self.per_episode_metrics = defaultdict(RunningMoments)
        tensorboard_dir = f"{self.runtime_args.output_dir}/tensorboard"
        self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)

    def forward_step(self, data, iteration=0):
        answers = data['answer']
        str_outputs = data["str_outputs"]
        eval_funcs = data["eval_func"]
        list_strs = list(zip(answers, str_outputs, eval_funcs))
        rewards = self.get_math_rule_reward(list_strs, is_eval=False)
        return {"math_rewards": rewards}

    def get_math_rule_reward(self, list_strs, is_eval):

        # Math reward
        reorder_list_strs, reorder_idx = [], []
        for idx, (answer, str_output, eval_func) in enumerate(list_strs):
            if eval_func == 'math_rule':
                reorder_list_strs.append((answer, str_output))
                reorder_idx.append(idx)
        reorder_rewards, success = self.math_rule_rm(reorder_list_strs)

        if is_eval:
            self.stats["eval_rewards/math_rule_reward_mean"] = np.mean(reorder_rewards)
            self.stats["eval_rewards/math_rule_parsing_rate"] = np.mean(success)
        else:
            self.stats["rewards/math_rule_reward_mean"] = np.mean(reorder_rewards)
            self.stats["rewards/math_rule_parsing_rate"] = np.mean(success)
        scores = [0] * len(list_strs)
        for idx, r in zip(reorder_idx, reorder_rewards):
            scores[idx] = r

        if is_eval:
            return scores
        else:
            self.per_episode_metrics["rewards/math_reward_model_scores"].update(torch.FloatTensor(scores))
            return scores

    def eval_forward(self, policy_res: dict):
        prompt_dicts = policy_res["prompt_dicts"]
        str_outputs = policy_res["str_outputs"]

        list_strs = [[prompt_dict, str_output] for prompt_dict, str_output in zip(prompt_dicts, str_outputs)]

        reward_model_scores = self.get_math_rule_reward(list_strs, is_eval=True)
        reward_model_scores = torch.FloatTensor(reward_model_scores)
        reward_model_scores = reward_model_scores.view(-1, 1)

        self.per_episode_metrics["eval_rewards/math_reward_model_scores"].update(reward_model_scores)

        reward_checkpoint = self.model_args['load']
        reward_checkpoint_load_iteration = self.model_args['load_iteration']

        output = []
        rewards_output = []
        for prompt_dict, str_output, reward in zip(prompt_dicts, str_outputs, reward_model_scores):
            rw = reward.cpu().item()
            rewards_output.append(rw)

            score_dict = {reward_checkpoint: {reward_checkpoint_load_iteration: [rw]}}
            j = {"query": prompt_dict, "responses": [str_output], "eval_score_dict": score_dict}
            output.append(j)
        self.log_each_step()

        return {"eval_jsonl": output, "rewards": rewards_output, "type": ["math_rule"] * len(rewards_output)}

    def log_each_step(self):
        stats_episode = self.stats
        stats_episode.update(get_running_stats(self.per_episode_metrics))

        stats_episode["exp_scores/running_math_mean"] = self.running.mean
        stats_episode["exp_scores/running_math_std"] = self.running.std

        print(f"score only/running_math_mean {self.running.mean}", flush=True)
        tensorboard_scalar_dict(self.tensorboard_writer, prefix=f"rewards_each/replica_id{self.replica_id}",
                                global_step=self._iteration,
                                scalar_dict=stats_episode)
        reset_running_stats(self.per_episode_metrics)
