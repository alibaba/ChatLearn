import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import chatlearn
from chatlearn.utils import future
from chatlearn import RLHFEngine
from chatlearn import TorchModule

from utils import CustomDataset, PolicyModel, ReferenceModel, ValueModel, PPOPolicy, PPOValue


class RewardModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reward forward =========", flush=True)
        data["reward_out"] = data["ref_out"] + data["policy_out"]
        return data


chatlearn.init()
chatlearn.get_args().models["policy"].num_replica = 2
policy = PolicyModel("policy")
reference = ReferenceModel("reference")
reward = RewardModel("reward")
value = ValueModel("value")
ppo_policy = PPOPolicy("ppo_policy")
ppo_value = PPOValue("ppo_value")

engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
#assert policy.num_replica == 2

data = torch.ones([1024])
engine.set_dataset([data] * 35)
engine.learn()
if policy.num_replica == 2:
    assert reference.num_replica == 1
    data = torch.ones([1024])
    assert len(engine.env._dataset) == 35, len(engine.env._dataset)
    visible_devices = engine.models[0].replicas[0].get_visible_gpus()
    visible_devices = future.get(visible_devices)
    assert visible_devices == [[0]], visible_devices
    visible_devices = engine.models[0].replicas[1].get_visible_gpus()
    visible_devices = future.get(visible_devices)
    assert visible_devices == [[1]], visible_devices
