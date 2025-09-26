"""compute advantage for grpo"""
from typing import List, Dict, Any
from collections import defaultdict

import numpy as np

from chatlearn.utils.utils import map_reduce_metrics


class AdvantageComputer:
    """advantage computer"""
    def __init__(self, num_inference_per_prompt):
        self._metric_prefix = 'advantage_compute'
        self._metric_list = []
        self.rule_reward_buffer = defaultdict(list)
        self.num_inference_per_prompt = num_inference_per_prompt

    def __call__(self, episode_replay_buffers: List[Dict[str, Any]]):
        buffers = episode_replay_buffers[-1].buffer
        all_correct_list = []
        all_wrong_list = []
        # Update buffer first
        for s in buffers:
            sample_id = s['prompt_uid']
            self.rule_reward_buffer[sample_id].append(s["rule_reward"])

        # Calculate advantage for all samples
        for s in buffers:
            sample_id = s['prompt_uid']
            is_all_zeros = all(x == 0 for x in self.rule_reward_buffer[sample_id])
            is_all_ones = all(x == 1 for x in self.rule_reward_buffer[sample_id])
            avg = np.mean(self.rule_reward_buffer[sample_id])
            std = np.std(self.rule_reward_buffer[sample_id])
            s['advantages'] = (s["rule_reward"] - avg) / (std + 1e-5)
            all_correct_list.append(is_all_ones)
            all_wrong_list.append(is_all_zeros)

        # clean buffer
        self.rule_reward_buffer = {k: v for k, v in self.rule_reward_buffer.items() if len(v) < self.num_inference_per_prompt}
        self.rule_reward_buffer = defaultdict(list, self.rule_reward_buffer)
        all_correct_ratio = sum(all_correct_list) / len(all_correct_list)
        all_wrong_ratio = sum(all_wrong_list) / len(all_wrong_list)
        self._metric_list.append({"all_correct_ratio": all_correct_ratio, "all_wrong_ratio": all_wrong_ratio})
        return buffers

    def get_and_clear_metrics(self):
        if self._metric_list is None or len(self._metric_list) == 0:
            return self._metric_prefix, {}

        reduced_metrics = map_reduce_metrics(self._metric_list)
        self._metric_list = []
        return self._metric_prefix, reduced_metrics
