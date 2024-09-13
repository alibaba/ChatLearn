from torch.utils.data import Dataset

import chatlearn
from chatlearn import EvalEngine, Evaluator
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

    def build_dataset(self, prompts, is_eval):
        dataset = CustomDataset(prompts)
        return dataset


class RewardModel(TorchModule):

    def eval_step(self, data):
        new_data = {}
        new_data['reward'] = ['reward_' + item for item in data['policy']]
        return new_data


class RewardModel2(TorchModule):

    def eval_step(self, data):
        new_data = {}
        new_data['reward2'] = ['reward2_' + item for item in data['policy']]
        return new_data


chatlearn.init()
chatlearn.get_args().models["policy"].num_gpu = 3
policy = PolicyModel("policy")
reward = RewardModel("reward")
reward2 = RewardModel2("reward2")


class CustomEngine(EvalEngine):

    def __init__(self, models):
        def eval_flow(batch):
            p = policy.forward_step(batch)
            r = reward.eval_step(p)
            r1 = reward2.eval_step(p)
            return r, r1
        evaluator = Evaluator(eval_flow)
        super().__init__(models, evaluator=evaluator)


engine = CustomEngine([policy, reward, reward2])

assert policy.num_replica == 3, policy.num_replica
train_prompts = ['query_' + str(i) for i in range(10, 91)]

engine.set_dataset(train_prompts)
results = engine.eval()
out = []
for model_name in ['reward', 'reward2']:
    for batch in results[model_name]:
        assert model_name in batch.keys()
    total = sum(len(each[model_name]) for each in results[model_name])
    assert total == len(train_prompts)
