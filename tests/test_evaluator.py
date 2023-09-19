import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import chatlearn
from chatlearn import RLHFEngine
from chatlearn import RLHFTorchModule

import chatlearn
from chatlearn import EvalEngine


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.collate_fn = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"query": self.data[idx]}



class PolicyModel(RLHFTorchModule):

    def setup(self):
        time.sleep(0.05)

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        return data

    def build_dataset(self, prompts):
        dataset = CustomDataset(prompts)
        return dataset


chatlearn.init()
chatlearn.get_args().models["policy"].num_device = 3
policy = PolicyModel("policy")
policy.register_eval_func("forward_step")
engine = EvalEngine(policy)

assert policy.num_replica == 3, policy.num_replica
train_prompts = ['query_'+str(i) for i in range(10, 91)]

engine.set_dataset(train_prompts)
results = engine.eval()

out = []
for value in results:
    out += value['query']
out = sorted(out)
assert out == train_prompts
