"""compute advantage for grpo"""
from collections import defaultdict

import torch

def compute_grpo_adv(episode_replay_buffers):
    buffers = episode_replay_buffers[-1].buffer
    queryids2samples = defaultdict(list)
    sample_id = 0
    for s in buffers:
        s['sample_id'] = sample_id
        queryids2samples[hash(",".join(map(str, s["prompt_token_ids"])))].append(s)
        sample_id += 1

    res_buffers = []
    for _, l in queryids2samples.items():
        rewards = [each["rule_rewards"] for each in l]
        rewards = torch.cat(rewards, dim=0)

        mean = torch.mean(rewards)
        std = torch.std(rewards)
        for i, li in enumerate(l):
            li["advantages"] = (rewards[i] - mean) / (std + 1e-5)
        res_buffers.extend(l)

    # Sort samples by original order in buffer
    res_buffers.sort(key=lambda x: x["sample_id"])
    for data in res_buffers:
        data.pop("sample_id")
    return res_buffers
