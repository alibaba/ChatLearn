import torch

import torch

import chatlearn
from chatlearn import RLHFEngine
from chatlearn.utils import future
from utils import PolicyModel, ReferenceModel, RewardModel, ValueModel, PPOPolicy, PPOValue

chatlearn.init()
chatlearn.get_args().models["policy"].num_gpu = 4
chatlearn.get_args().models["policy"].tensor_model_parallel_size = 2
chatlearn.get_args().models["reference"].num_gpu = 4
chatlearn.get_args().models["reference"].tensor_model_parallel_size = 2
chatlearn.get_args().models["policy"].gpu_per_process = 1
chatlearn.get_args().models["reference"].gpu_per_process = 1
chatlearn.get_args().runtime_args.colocation = [["policy", "reference"]]

chatlearn.get_args().models["policy"].num_replica = 1
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
engine.set_dataset(data)

engine.setup()
a = torch.ones([1])
b = torch.ones([1])
model = engine.models[0]
model2 = engine.models[1]

for replica_id in range(len(model.replicas)):
    visible_devices = future.get(model.replicas[replica_id].get_visible_gpus())
    if replica_id == 0:
        assert visible_devices == [[0], [1]], visible_devices
    else:
        assert visible_devices == [[2], [3]], visible_devices
    print(visible_devices)
    visible_devices = future.get(model2.replicas[replica_id].get_visible_gpus())
    if replica_id == 0:
        assert visible_devices == [[0], [1]], visible_devices
    else:
        assert visible_devices == [[2], [3]], visible_devices
    print(visible_devices)
engine.env.setup()

def check_output_models(model_name, expected_models):
    assert [node.model.name for node in engine.env.model_flow.get(model_name).output_models] == expected_models

check_output_models("policy", ['reference', 'value', 'reward'])
check_output_models("reference", ['reward'])
check_output_models("value", ['reward'])
check_output_models("reward", [])

def check_colocate_models(model_name, expected_models):
    assert [model.name for model in engine.env.model_flow.get(model_name).model.colocate_models] == expected_models

check_colocate_models("policy", ['reference'])
check_colocate_models("reference", ['policy'])
