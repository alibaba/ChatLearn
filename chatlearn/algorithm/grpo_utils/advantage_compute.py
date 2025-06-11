from collections import defaultdict

import torch

def compute_grpo_adv(episode_replay_buffers):
    buffers = episode_replay_buffers[-1].buffer
    queryids2samples = defaultdict(list)
    for s in buffers:
        queryids2samples[hash(','.join(map(str, s["prompt_token_ids"])))].append(s)
    
    res_buffers = []
    for _, l in queryids2samples.items():
        rewards = [each["rule_rewards"] for each in l]
        rewards = torch.cat(rewards, dim=0)

        mean = torch.mean(rewards)
        std = torch.std(rewards)
        for i, li in enumerate(l):
            li['advantages'] = ((rewards[i] - mean) / (std + 1e-5))
        res_buffers.extend(l)
    return res_buffers