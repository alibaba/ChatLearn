import torch

import torch

import chatlearn
from chatlearn import RLHFEngine, Evaluator
from chatlearn.utils import future
from utils import PolicyModel, ReferenceModel, RewardModel, ValueModel, PPOPolicy, PPOValue


def test_rlhf_placement_colocate():
    chatlearn.get_args().models["policy"].num_gpu = 4
    chatlearn.get_args().models["policy"].tensor_model_parallel_size = 2
    chatlearn.get_args().models["reference"].num_gpu = 4
    chatlearn.get_args().models["reference"].tensor_model_parallel_size = 2
    chatlearn.get_args().models["policy"].gpu_per_process = 1
    chatlearn.get_args().models["reference"].gpu_per_process = 1

    # Reset to 1, because we don't know what's the configuration of last case
    # And we don't want re-init the whole chatlearn framework either
    chatlearn.get_args().models["reward"].num_gpu = 1
    chatlearn.get_args().models["value"].num_gpu = 1
    chatlearn.get_args().models["ppo_value"].num_gpu = 1
    chatlearn.get_args().models["ppo_value"].tensor_model_parallel_size = 1
    chatlearn.get_args().models["ppo_policy"].num_gpu = 1
    chatlearn.get_args().models["ppo_policy"].tensor_model_parallel_size = 1

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
        model_node = [node for node in engine.env.model_flow.model_nodes if node.name == model_name][0]
        assert [node.model.name for node in model_node.output_nodes] == expected_models

    check_output_models("policy", ['reference', 'value', 'reward'])
    check_output_models("reference", ['reward'])
    check_output_models("value", ['reward'])
    check_output_models("reward", [])

    def check_colocate_models(model_name, expected_models):
        model_node = [node for node in engine.env.model_flow.model_nodes if node.name == model_name][0]
        assert [model.name for model in model_node.model.colocate_models] == expected_models

    check_colocate_models("policy", ['reference'])
    check_colocate_models("reference", ['policy'])
    engine.stop()

def test_rlhf_placement_colocate_2():
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
        model_node = [node for node in engine.env.model_flow.model_nodes if node.name == model_name][0]
        assert [node.model.name for node in model_node.output_nodes] == expected_models

    check_output_models("policy", ['reference', 'value', 'reward'])
    check_output_models("reference", ['reward'])
    check_output_models("value", ['reward'])
    check_output_models("reward", [])


    def check_colocate_models(model_name, expected_models):
        model_node = [node for node in engine.env.model_flow.model_nodes if node.name == model_name][0]
        assert [model.name for model in model_node.model.colocate_models] == expected_models

    check_colocate_models("policy", ['reference', 'value', 'reward', 'ppo_policy', 'ppo_value'])
    check_colocate_models("reference", ['policy', 'reward', 'ppo_policy', 'ppo_value'])
    check_colocate_models("value", ['policy', 'reward', 'ppo_policy', 'ppo_value'])
    check_colocate_models("reward", ['policy', 'reference', 'value', 'ppo_policy', 'ppo_value'])

    def check_next_colocate_model(model_name, expected_model):
        model_node = [node for node in engine.env.model_flow.model_nodes if node.name == model_name][0]
        assert model_node.next_colocate_node.name == expected_model

    check_next_colocate_model("policy", "reference")
    check_next_colocate_model("reference", "reward")
    check_next_colocate_model("value", "reward")
    check_next_colocate_model("value", "reward")
    engine.stop()

TEST_CASE = [test_rlhf_placement_colocate, test_rlhf_placement_colocate_2]