from torch.utils.data import Dataset
import chatlearn
from chatlearn import EvalEngine
from chatlearn import TorchModule


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
        new_data = {}
        new_data['policy'] = ['policy_' + item for item in data['query']]
        return new_data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset


class RewardModel(TorchModule):

    def eval_step(self, data):
        new_data = {}
        new_data['reward'] = ['reward_' + item for item in data['policy']]
        return new_data


chatlearn.init()
chatlearn.get_args().models["policy"].num_gpu = 3
policy = PolicyModel("policy")

reward = RewardModel("reward")

def eval_flow(b):
    r0 = policy.forward_step(b)
    r1 = reward.eval_step(r0)
    return r1

engine = EvalEngine(eval_flow)

assert policy.num_replica == 3, policy.num_replica
train_prompts = ['query_' + str(i) for i in range(10, 91)]

engine.set_dataset(train_prompts)
results = engine.eval()['reward']
out = []
for value in results:
    out += value['reward']
out = sorted(out)
assert out == [f"reward_policy_{p}" for p in train_prompts]
