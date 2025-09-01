"""compute advantage for grpo"""
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np

def compute_grpo_adv(episode_replay_buffers: List[Dict[str, Any]]):
    buffers = episode_replay_buffers[-1].buffer
    queryids2samples = defaultdict(list)
    sample_id = 0
    for s in buffers:
        s["advantages"] = (s["rule_reward"] - s["rule_reward_mean"]) / (s["rule_reward_std"] + 1e-5)
    return buffers
