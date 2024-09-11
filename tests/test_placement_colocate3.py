import time

import torch

import chatlearn
from chatlearn.utils import future
from chatlearn import RLHFEngine
from chatlearn import TorchModule


chatlearn.init()

def set_model(name, tp, gpu_per_process, num_gpu):
    print(chatlearn.get_args().models.keys())
    chatlearn.get_args().models[name].num_gpu = num_gpu
    chatlearn.get_args().models[name].gpu_per_process = gpu_per_process
    chatlearn.get_args().models[name].tensor_model_parallel_size = tp

set_model("policy", 4, 1, 8)
set_model("value", 1, 1, 1)
set_model("reward", 1, 1, 4)
set_model("reference", 4, 1, 8)
set_model("ppo_policy", 8, 1, 8)
set_model("ppo_value", 8, 1, 8)

chatlearn.get_args().runtime_args.colocation = [["policy", "reference", "reward", "value", "ppo_policy", "ppo_value"]]
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.collate_fn = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"query": self.data[idx]}


class PolicyModel(TorchModule):

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        query = data["query"]
        self.put("policy_put", 100)
        data["policy_out"] = query
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset


class ReferenceModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reference forward =========", flush=True)
        query = data["policy_out"].cuda()
        data["ref_out"] = query
        return data


class RewardModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reward forward =========", flush=True)
        data["reward_out"] = data["ref_out"].cuda() + data["policy_out"].cuda()
        policy_put = self.get("policy_put")
        assert policy_put == 100
        return data


class ValueModel(TorchModule):

    def forward_step(self, data, iteration):
        print("value forward =========", flush=True)
        data["value_out"] = data["policy_out"].cuda() * 3
        return data


class PPOPolicy(TorchModule):

    def train_step(self, data, iteration):
        print("ppo policy train_step =========", flush=True)
        num_mb = len(data)
        return num_mb


class PPOValue(TorchModule):

    def train_step(self, data, iteration):
        print("ppo value train_step =========", flush=True)
        num_mb = len(data)
        return num_mb


policy = PolicyModel("policy")
reference = ReferenceModel("reference")
reward = RewardModel("reward")
value = ValueModel("value")
ppo_policy = PPOPolicy("ppo_policy")
ppo_value = PPOValue("ppo_value")

engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
data = torch.ones([1024])
engine.set_dataset([data] * 35)
engine.setup()

for replica_id in range(len(engine.named_models['ppo_policy'].replicas)):
    visible_devices = future.get(engine.named_models['ppo_policy'].replicas[replica_id].get_visible_gpus())
    if replica_id == 0:
        assert visible_devices == [[0], [1], [2], [3], [4], [5], [6], [7]], visible_devices

for replica_id in range(len(engine.named_models['policy'].replicas)):
    visible_devices = future.get(engine.named_models['policy'].replicas[replica_id].get_visible_gpus())
    if replica_id == 0:
        assert visible_devices == [[3], [2], [1], [0]], visible_devices
    else:
        assert visible_devices == [[7], [6], [5], [4]], visible_devices

for replica_id in range(len(engine.named_models['reference'].replicas)):
    visible_devices = future.get(engine.named_models['reference'].replicas[replica_id].get_visible_gpus())
    if replica_id == 0:
        assert visible_devices == [[0], [1], [2], [3]], visible_devices
    else:
        assert visible_devices == [[4], [5], [6], [7]], visible_devices

for replica_id in range(len(engine.named_models['value'].replicas)):
    visible_devices = future.get(engine.named_models['value'].replicas[replica_id].get_visible_gpus())
    assert visible_devices[0][0] == replica_id+4, visible_devices

for replica_id in range(len(engine.named_models['reward'].replicas)):
    visible_devices = future.get(engine.named_models['reward'].replicas[replica_id].get_visible_gpus())
    assert visible_devices[0][0] == replica_id, visible_devices
