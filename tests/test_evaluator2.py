import time

from torch.utils.data import Dataset

import chatlearn
from chatlearn import EvalEngine
from chatlearn import RLHFTorchModule


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
        new_data = {}
        new_data['policy'] = ['policy_' + item for item in data['query']]
        return new_data

    def build_dataset(self, prompts):
        dataset = CustomDataset(prompts)
        return dataset


class RewardModel(RLHFTorchModule):

    def setup(self):
        time.sleep(0.05)

    def eval_step(self, data):
        new_data = {}
        new_data['reward'] = ['reward_' + item for item in data['policy']]
        return new_data


chatlearn.init()
chatlearn.get_args().models["policy"].num_device = 3
policy = PolicyModel("policy")
policy.register_eval_func("forward_step")

reward = RewardModel("reward")
reward.register_eval_func("eval_step")
engine = EvalEngine([policy, reward])

assert policy.num_replica == 3, policy.num_replica
train_prompts = ['query_' + str(i) for i in range(10, 91)]

engine.set_dataset(train_prompts)
results = engine.eval()
out = []
for value in results:
    out += value['reward']
out = sorted(out)
assert out == [f"reward_policy_{p}" for p in train_prompts]
