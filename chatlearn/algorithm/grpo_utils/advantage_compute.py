"""compute advantage for grpo"""
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np

def compute_grpo_adv(episode_replay_buffers: List[Dict[str, Any]]):
    buffers = episode_replay_buffers[-1].buffer
    queryids2samples = defaultdict(list)
    sample_id = 0
    for s in buffers:
        s['sample_id'] = sample_id
        queryids2samples[hash(",".join(map(str, s["prompt_token_ids"])))].append(s)
        sample_id += 1

    res_buffers = []
    # TODO: torch and numpy have difference result, not knowing consequence
    for _, l in queryids2samples.items():
        rewards = np.array([each["rule_reward"] for each in l])
        mean = np.mean(rewards)
        std = np.std(rewards)

        for li in l:
            li["advantages"] = (li["rule_reward"] - mean) / (std + 1e-5)
        res_buffers.extend(l)

    # Sort samples by original order in buffer
    res_buffers.sort(key=lambda x: x["sample_id"])
    for data in res_buffers:
        data.pop("sample_id")
    return res_buffers
