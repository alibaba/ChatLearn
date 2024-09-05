import torch

import torch

import chatlearn
from chatlearn import RLHFEngine, Evaluator
from utils import PolicyModel, ReferenceModel, RewardModel, ValueModel, PPOPolicy, PPOValue


chatlearn.init()
chatlearn.get_args().models["policy"].num_gpu = 8
chatlearn.get_args().models["reference"].num_gpu = 4
chatlearn.get_args().models["value"].num_gpu = 4
chatlearn.get_args().models["reward"].num_gpu = 8
chatlearn.get_args().models["ppo_policy"].num_gpu = 8
chatlearn.get_args().models["ppo_value"].num_gpu = 8
chatlearn.get_args().runtime_args.colocation = [["policy", "reference", "value", "reward", "ppo_policy", "ppo_value"]]

policy = PolicyModel("policy")
reference = ReferenceModel("reference")
reward = RewardModel("reward")
value = ValueModel("value")
ppo_policy = PPOPolicy("ppo_policy")
ppo_value = PPOValue("ppo_value")

engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
data = []
for i in range(35):
    data.append(torch.ones([10]) * i)

def eval_flow(batch):
    p0 = policy.forward_step(batch)
    r0 = reward.forward_step(p0)
    return r0

evaluator = Evaluator(eval_flow).set_dataset(data)
engine.set_evaluator(evaluator)
engine.set_dataset(data)

engine.setup()

engine.env.setup()
engine.evaluator.setup()

def check_output_models(model_name, expected_models):
    assert [node.model.name for node in engine.env.model_flow.get(model_name).output_models] == expected_models

check_output_models("policy", ['reference', 'value', 'reward'])
check_output_models("reference", ['reward'])
check_output_models("value", ['reward'])
check_output_models("reward", [])

def check_colocate_models(model_name, expected_models):
    assert [model.name for model in engine.env.model_flow.get(model_name).model.colocate_models] == expected_models

check_colocate_models("policy", ['reference', 'value', 'reward', 'ppo_policy', 'ppo_value'])
check_colocate_models("reference", ['policy', 'reward', 'ppo_policy', 'ppo_value'])
check_colocate_models("value", ['policy', 'reward', 'ppo_policy', 'ppo_value'])
check_colocate_models("reward", ['policy', 'reference', 'value', 'ppo_policy', 'ppo_value'])

def check_next_colocate_model(model_name, expected_model):
    assert engine.env.model_flow.get(model_name).next_colocate_node.name == expected_model

check_next_colocate_model("policy", "reference")
check_next_colocate_model("reference", "reward")
check_next_colocate_model("value", "reward")
check_next_colocate_model("value", "reward")
