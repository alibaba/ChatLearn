import torch
from torch.utils.data import Dataset

import chatlearn
from chatlearn import Engine
from chatlearn import EvalEngine
from chatlearn import TorchModule
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.trainer import Trainer

from utils import CustomDataset, RewardModel, ValueModel, PPOPolicy, PPOValue


class PolicyModel(TorchModule):

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


chatlearn.init()
policy = PolicyModel("policy")
reference = ReferenceModel("reference")

def env_compute_flow(batch):
    policy_out = policy.forward_step(batch)
    ref_out = reference.forward_step(policy_out, batch)
    return ref_out

engine = EvalEngine(env_compute_flow)

assert policy.num_replica == 1
assert reference.num_replica == 1
data = torch.ones([1024])
engine.set_dataset([data] * 35)

engine.eval()
assert len(engine._dataset) == 35, len(engine._dataset)
