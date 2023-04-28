import torch
from rlhf.engine import RLHFEngine
from rlhf.model_wrapper import RLHFTorchModule
import time
import rlhf


rlhf.init()

def set_model(name, num_device, gpu_per_process, num_replica):
    print(rlhf.get_args().models.keys())
    rlhf.get_args().models[name].num_device = num_device
    rlhf.get_args().models[name].gpu_per_process = gpu_per_process
    rlhf.get_args().models[name].num_replica = num_replica

set_model("policy", 4, 1, 2)
set_model("value", 1, 1, 4)
set_model("reward", 1, 1, 4)
set_model("reference", 4, 1, 2)
set_model("ppo_policy", 8, 1, 1)
set_model("ppo_value", 8, 1, 1)

rlhf.get_args().rlhf_args.num_rollout_worker = 1
rlhf.get_args().rlhf_args.colocation = [["policy", "reference", "reward", "value", "ppo_policy", "ppo_value"]]
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



for replica_id in range(len(engine.ppo_policy.replicas)):
    visible_devices = rlhf.get(engine.ppo_policy.replicas[replica_id].get_visible_gpus())
    if replica_id == 0:
        assert visible_devices == [[0], [1], [2], [3], [4], [5], [6], [7]], visible_devices

for replica_id in range(len(engine.policy.replicas)):
    visible_devices = rlhf.get(engine.policy.replicas[replica_id].get_visible_gpus())
    if replica_id == 0:
        assert visible_devices == [[3], [2], [1], [0]], visible_devices
    else:
        assert visible_devices == [[7], [6], [5], [4]], visible_devices

for replica_id in range(len(engine.reference.replicas)):
    visible_devices = rlhf.get(engine.reference.replicas[replica_id].get_visible_gpus())
    if replica_id == 0:
        assert visible_devices == [[0], [1], [2], [3]], visible_devices
    else:
        assert visible_devices == [[4], [5], [6], [7]], visible_devices

for replica_id in range(len(engine.value.replicas)):
    visible_devices = rlhf.get(engine.value.replicas[replica_id].get_visible_gpus())
    assert visible_devices[0][0] == replica_id+4, visible_devices

for replica_id in range(len(engine.reward.replicas)):
    visible_devices = rlhf.get(engine.reward.replicas[replica_id].get_visible_gpus())
    assert visible_devices[0][0] == replica_id, visible_devices


