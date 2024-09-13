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

set_model("policy", 4, 1, 4)
set_model("value", 1, 1, 2)
set_model("reward", 1, 1, 4)
set_model("reference", 2, 1, 2)
set_model("ppo_policy", 4, 1, 4)
set_model("ppo_value", 4, 1, 4)

chatlearn.get_args().runtime_args.colocation = [["policy", "ppo_policy", "ppo_value"], ["reference", "reward", "value"]]
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
engine.learn()

def check_output_models(model_name, expected_models):
    assert [node.model.name for node in engine.env.model_flow.get(model_name).output_models] == expected_models

check_output_models("policy", ['reference', 'value', 'reward'])
check_output_models("reference", ['reward'])
check_output_models("value", ['reward'])
check_output_models("reward", [])

def check_colocate_models(model_name, expected_models):
    assert [model.name for model in engine.env.model_flow.get(model_name).model.colocate_models] == expected_models

check_colocate_models("policy", ['ppo_policy', 'ppo_value'])
check_colocate_models("reference", ['reward'])
check_colocate_models("value", ['reward'])
check_colocate_models("reward", ['reference', 'value'])

def check_next_colocate_model(model_name, expected_model):
    if engine.env.model_flow.get(model_name).next_colocate_node:
        assert engine.env.model_flow.get(model_name).next_colocate_node.name == expected_model
    else:
        assert engine.env.model_flow.get(model_name).next_colocate_node is expected_model

check_next_colocate_model("policy", None)
check_next_colocate_model("reference", "reward")
check_next_colocate_model("value", "reward")
check_next_colocate_model("reward", None)
