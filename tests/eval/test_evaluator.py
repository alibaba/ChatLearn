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


def test_evaluator():
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
    engine.stop()



def test_evaluator_2():
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
    engine.stop()


def test_evaluator_3():
    chatlearn.get_args().models["policy"].num_replica = 2
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)

    data = torch.ones([1024])
    data = [data] * 40
    for i in range(40):
        data[i] = data[i] * i
    train_data = data[:32]
    val_data = data[32:]

    def eval_flow(batch):
        r0 = policy.forward_step(batch)
        r1 = reward.forward_step(r0)
        return r1

    def eval_post_process(results, eval_info):
        results = results["reward"]
        results = listdict_to_dictlist(results)
        eval_num = len(results['reward_out']) * results['reward_out'][0].shape[0]
        assert eval_num == chatlearn.get_args().runtime_args.get('eval_data_num_limit'), \
            f"expect the number of evaluated samples is equal to eval_data_num_limit, but get {eval_num} " \
            f"and {chatlearn.get_args().runtime_args.get('eval_data_num_limit')} respectively"
        assert torch.min(results['reward_out'][0]) == 32
        assert torch.max(results['reward_out'][0]) == 33
        assert torch.min(results['reward_out'][1]) == 34
        assert torch.max(results['reward_out'][1]) == 35

    eval_num_limit = chatlearn.get_args().runtime_args.get('eval_data_num_limit')
    eval_num_limit = min(eval_num_limit, len(val_data))
    val_data = val_data[:eval_num_limit]
    evaluator = Evaluator(eval_flow).set_dataset(val_data).set_post_process_func(eval_post_process)
    engine.set_evaluator(evaluator)

    engine.set_dataset(train_data)
    engine.setup()
    for executor in engine._executors:
        if executor:
            executor.setup()

    engine.evaluate(0)
    engine.stop()

TEST_CASE = [test_evaluator, test_evaluator_2, test_evaluator_3]