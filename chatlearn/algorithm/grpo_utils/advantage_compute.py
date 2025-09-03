"""compute advantage for grpo"""
from typing import List, Dict, Any
from collections import defaultdict

import numpy as np


class AdvantageComputer:
    """advantage computer"""
    def __init__(self, num_inference_per_prompt):
        self.rule_reward_buffer = defaultdict(list)
        self.num_inference_per_prompt = num_inference_per_prompt

    def __call__(self, episode_replay_buffers: List[Dict[str, Any]]):
        buffers = episode_replay_buffers[-1].buffer
        # Update buffer first
        for s in buffers:
            sample_id = s['prompt_uid']
            self.rule_reward_buffer[sample_id].append(s["rule_reward"])

        # Calculate advantage for all samples
        for s in buffers:
            sample_id = s['prompt_uid']
            avg = np.mean(self.rule_reward_buffer[sample_id])
            std = np.std(self.rule_reward_buffer[sample_id])
            s['advantages'] = (s["rule_reward"] - avg) / (std + 1e-5)

        # clean buffer
        pop_keys = [k for k,v in self.rule_reward_buffer.items() if len(v) == self.num_inference_per_prompt]
        for key in pop_keys:
            self.rule_reward_buffer.pop(key)
        return buffers
