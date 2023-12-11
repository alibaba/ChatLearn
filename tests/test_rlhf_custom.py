import time

import torch
from torch.utils.data import Dataset

import chatlearn
from chatlearn import Engine
from chatlearn import RLHFTorchModule
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.trainer import Trainer


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.collate_fn = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"query": self.data[idx]}


chatlearn.init()


class PolicyModel(RLHFTorchModule):

    def setup(self):
        time.sleep(0.05)

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        query = data["query"]
        bs = query.size(0)
        data["policy_out"] = torch.ones([bs, 1024]).cuda()
        return data

    def build_dataset(self, prompts):
        dataset = CustomDataset(prompts)
        return dataset


class ReferenceModel(RLHFTorchModule):

    def forward_step(self, data, iteration):
        print("reference forward =========", flush=True)
        query = data["policy_out"].cuda()
        data["ref_out"] = query * 2
        return data


class RewardModel(RLHFTorchModule):

    def forward_step(self, data, iteration):
        print("reward forward =========", flush=True)
        data["reward_out"] = data["ref_out"].cuda() + data["policy_out"].cuda()
        return data


class ValueModel(RLHFTorchModule):

    def forward_step(self, data, iteration):
        print("value forward =========", flush=True)
        data["value_out"] = data["policy_out"].cuda() * 3
        return data


class PPOPolicy(RLHFTorchModule):

    def train_step(self, data, train_info):
        print("ppo policy train_step =========", flush=True)
        num_mb = len(data)
        return num_mb


class PPOValue(RLHFTorchModule):

    def train_step(self, data, train_info):
        print("ppo value train_step =========", flush=True)
        num_mb = len(data)
        return num_mb


policy = PolicyModel("policy")
reference = ReferenceModel("reference")
reward = RewardModel("reward")
value = ValueModel("value")
ppo_policy = PPOPolicy("ppo_policy")
ppo_value = PPOValue("ppo_value")


def env_compute_flow(batch):
    policy_out = policy.forward_step(batch)
    ref_out = reference.forward_step(policy_out)
    value_out = value.forward_step(policy_out)
    reward_out = reward.forward_step(policy_out, ref_out, value_out)
    return value_out, reward_out


def trainer_compute_flow(batch):
    ppo_policy.train_step(batch)
    ppo_value.train_step(batch)


env = Environment([policy, value, reference, reward]).set_flow(env_compute_flow)
trainer = Trainer([ppo_policy, ppo_value]).set_flow(trainer_compute_flow)

engine = Engine(env, trainer)
engine.set_parameter_sync(ppo_policy, policy)
engine.set_parameter_sync(ppo_value, value)
assert policy.num_replica == 1
assert reference.num_replica == 1
data = torch.ones([1024])
engine.set_dataset([data] * 35)

engine.learn()
assert len(engine.env._dataset) == 35, len(engine.env._dataset)
