import torch
import rlhf
from rlhf.engine import RLHFEngine
from rlhf.model_wrapper import RLHFTorchModule
import time
import rlhf
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

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
rlhf.get_args().rlhf_args.num_rollout_worker = 2
rlhf.get_args().rlhf_args.colocation = [["reference", "reward"]]

policy = PolicyModel("policy")
reference = ReferenceModel("reference")
reward = RewardModel("reward")
value = ValueModel("value")
ppo_policy = PPOPolicy("ppo_policy")
ppo_value = PPOValue("ppo_value")


engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
assert policy.num_replica == 2
assert reward.num_replica == 2
assert reference.num_replica == 2
data = torch.ones([1024])
engine.set_dataset([data] * 35)
assert len(engine.env._dataset[0]) == 20, len(engine.env._dataset[0])
assert len(engine.env._dataset[1]) == 20, len(engine.env._dataset[0])
engine.set_dataset([data] * 35, drop_last=True)
assert len(engine.env._dataset[0]) == 16, len(engine.env._dataset[0])
assert len(engine.env._dataset[0]) == 16, len(engine.env._dataset[0])
#visible_devices = engine.models[0].replicas[0].get_visible_gpus()
for model in engine.models:
    for replica in model.replicas:
        print(model.name, replica, rlhf.get(replica.get_visible_gpus()), "====", flush=True)

data = torch.ones([1024])
engine.set_dataset([data] * 35)
engine.learn()
assert engine.episode_stats["episode"] < 3200
assert engine.episode_stats["episode"] > 2000
