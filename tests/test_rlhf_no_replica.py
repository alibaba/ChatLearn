import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import rlhf
from rlhf import RLHFEngine
from rlhf import RLHFTorchModule


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"query": self.data[idx]}




class PolicyModel(RLHFTorchModule):

    def setup(self):
        time.sleep(0.05)

    def forward_step(self, data):
        print("policy forward =========", flush=True)
        query = data["query"]
        time.sleep(1)
        data["policy_out"] = query
        return data

    def build_dataloader(self, prompts):
        return DataLoader(CustomDataset(prompts), batch_size=4, shuffle=True)



class ReferenceModel(RLHFTorchModule):


    def forward_step(self, data):
        print("reference forward =========", flush=True)
        query = data["policy_out"].cuda()
        time.sleep(0.01)
        data["ref_out"] = query
        return data


class RewardModel(RLHFTorchModule):


    def forward_step(self, data):
        print("reward forward =========", flush=True)
        data["reward_out"] = data["ref_out"].cuda() + data["policy_out"].cuda()
        time.sleep(0.01)
        return data

class ValueModel(RLHFTorchModule):

    def forward_step(self, data):
        print("value forward =========", flush=True)
        data["value_out"] = data["policy_out"].cuda() * 3
        time.sleep(0.01)
        return data


class PPOPolicy(RLHFTorchModule):

    def train_step(self, data, train_info):
        print("ppo policy train_step =========", flush=True)
        num_mb = len(data)
        time.sleep(0.1)
        return num_mb


class PPOValue(RLHFTorchModule):

    def train_step(self, data, train_info):
        print("ppo value train_step =========", flush=True)
        num_mb = len(data)
        time.sleep(0.1)
        return num_mb


rlhf.init()
rlhf.get_args().models["policy"].num_replica = 1
policy = PolicyModel("policy")
reference = ReferenceModel("reference")
reward = RewardModel("reward")
value = ValueModel("value")
ppo_policy = PPOPolicy("ppo_policy")
ppo_value = PPOValue("ppo_value")


engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)

data = torch.ones([1024])
engine.set_dataset([data] * 35)
engine.learn()

