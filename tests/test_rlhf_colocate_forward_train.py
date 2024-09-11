import time

import torch
from torch.utils.data import Dataset

import chatlearn
from chatlearn import Engine
from chatlearn import TorchModule
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.trainer import Trainer
from utils import CustomDataset, PolicyModel, ReferenceModel, RewardModel, PPOPolicy


class ValueModel(TorchModule):

    def forward_step(self, data, iteration):
        print("value forward =========", flush=True)
        data["value_out"] = data["policy_out"].cuda() * 3
        return data

    def train_step(self, data, iteration):
        print("ppo value train_step =========", flush=True)
        num_mb = len(data)
        return num_mb


chatlearn.init()
policy = PolicyModel("policy")
reference = ReferenceModel("reference")
reward = RewardModel("reward")
value = ValueModel("value")
ppo_policy = PPOPolicy("ppo_policy")

def env_compute_flow(batch):
    policy_out = policy.forward_step(batch)
    ref_out = reference.forward_step(policy_out)
    value_out = value.forward_step(policy_out)
    reward_out = reward.forward_step(policy_out, ref_out, value_out)
    return value_out, reward_out

def trainer_compute_flow(batch):
    ppo_policy.train_step(batch)
    value.train_step(batch)

env = Environment(env_compute_flow)
trainer = Trainer(trainer_compute_flow)

engine = Engine(env, trainer)
engine.set_parameter_sync(ppo_policy, policy)
assert policy.num_replica == 1
assert reference.num_replica == 1
data = torch.ones([1024])
engine.set_dataset([data] * 35)

engine.learn()
assert len(engine.env._dataset) == 35, len(engine.env._dataset)
