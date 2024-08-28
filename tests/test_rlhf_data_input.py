import time

import torch
from torch.utils.data import Dataset

import chatlearn
from chatlearn import Engine
from chatlearn import TorchModule
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.trainer import Trainer

from utils import CustomDataset, RewardModel, ValueModel, PPOPolicy, PPOValue


chatlearn.init()


class PolicyModel(TorchModule):

    def setup(self):
        time.sleep(0.05)

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        query = data["query"]
        bs = query.size(0)
        return {"policy_out": torch.ones([bs, 1024]).cuda()}

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset


class ReferenceModel(TorchModule):

    def forward_step(self, data, iteration):
        assert "query" in data.keys()
        query = data["policy_out"].cuda()
        data["ref_out"] = query * 2
        return data


policy = PolicyModel("policy")
reference = ReferenceModel("reference")
reward = RewardModel("reward")
value = ValueModel("value")
ppo_policy = PPOPolicy("ppo_policy")
ppo_value = PPOValue("ppo_value")


def env_compute_flow(batch):
    policy_out = policy.forward_step(batch)
    ref_out = reference.forward_step(policy_out, batch)
    value_out = value.forward_step(policy_out)
    reward_out = reward.forward_step(policy_out, ref_out, value_out)
    return value_out, reward_out


def trainer_compute_flow(batch):
    ppo_policy.train_step(batch)
    ppo_value.train_step(batch)


env = Environment(env_compute_flow)
trainer = Trainer(trainer_compute_flow)

engine = Engine(env, trainer)
engine.set_parameter_sync(ppo_policy, policy)
engine.set_parameter_sync(ppo_value, value)
assert policy.num_replica == 1
assert reference.num_replica == 1
data = torch.ones([1024])
engine.set_dataset([data] * 35)

engine.learn()
assert len(engine.env._dataset) == 35, len(engine.env._dataset)
