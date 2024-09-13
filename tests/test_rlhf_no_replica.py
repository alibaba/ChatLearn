import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import chatlearn
from chatlearn import RLHFEngine
from chatlearn import TorchModule

from utils import CustomDataset, PolicyModel, ReferenceModel, RewardModel, ValueModel, PPOPolicy, PPOValue


chatlearn.init()
chatlearn.get_args().models["policy"].num_replica = 1
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
