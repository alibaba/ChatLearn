import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import chatlearn
from chatlearn import RLHFEngine
from chatlearn import TorchModule

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


class PolicyModel(TorchModule):

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset


chatlearn.init()
chatlearn.get_args().models["policy"].num_gpu = 3
policy = PolicyModel("policy")

def eval_flow(b):
    r0 = policy.forward_step(b)
    return r0

engine = EvalEngine(eval_flow)

assert policy.num_replica == 3, policy.num_replica
train_prompts = ['query_'+str(i) for i in range(10, 91)]

engine.set_dataset(train_prompts)
results = engine.eval()["policy"]

out = []
for value in results:
    out += value['query']
out = sorted(out)
assert out == train_prompts
