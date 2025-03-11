import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import chatlearn
from chatlearn import RLHFEngine
from chatlearn import TorchModule
from chatlearn import Evaluator

from utils import PolicyModel, ReferenceModel, ValueModel, PPOPolicy, PPOValue, listdict_to_dictlist

class RewardModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reward forward =========", flush=True)
        data["reward_out"] = data["policy_out"].cuda()
        return data

chatlearn.init()
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

class CustomEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post_process(self, results, eval_info):
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
        return results

eval_num_limit = chatlearn.get_args().runtime_args.get('eval_data_num_limit')
eval_num_limit = min(eval_num_limit, len(val_data))
val_data = val_data[:eval_num_limit]
evaluator = CustomEvaluator(eval_flow).set_dataset(val_data)
engine.set_evaluator(evaluator)

engine.set_dataset(train_data)
engine.setup()
for executor in engine._executors:
    if executor:
        executor.setup()

engine.evaluate(0)
