import torch

import chatlearn
from chatlearn.utils import future
from chatlearn import RLHFEngine
from chatlearn import TorchModule


from torch.utils.data import DataLoader
from torch.utils.data import Dataset


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
        query = data["query"]
        data["policy_out"] = query
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset


class ReferenceModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reference forward =========", flush=True)
        query = data["policy_out"].cuda()
        data["ref_out"] = query
        return data


class RewardModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reward forward =========", flush=True)
        data["reward_out"] = data["ref_out"].cuda() + data["policy_out"].cuda()
        policy_put = self.get("policy_put")

        return data


class ValueModel(TorchModule):

    def forward_step(self, data, iteration):
        print("value forward =========", flush=True)
        data["value_out"] = data["policy_out"].cuda() * 3
        return data


class PPOPolicy(TorchModule):

    def train_step(self, data, iteration):
        print("ppo policy train_step =========", flush=True)
        num_mb = len(data)
        return num_mb


class PPOValue(TorchModule):

    def train_step(self, data, iteration):
        print("ppo value train_step =========", flush=True)
        num_mb = len(data)
        return num_mb



def set_model(name, tp, gpu_per_process, num_gpu):
        print(chatlearn.get_args().models.keys())
        chatlearn.get_args().models[name].num_gpu = num_gpu
        chatlearn.get_args().models[name].gpu_per_process = gpu_per_process
        chatlearn.get_args().models[name].tensor_model_parallel_size = tp

def _create_engine():
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
    data = torch.ones([1024])
    engine.set_dataset([data] * 35)
    engine.setup()
    return engine

def test_placement_colocate_2():
    set_model("policy", 4, 1, 8)
    set_model("value", 1, 1, 4)
    set_model("reward", 1, 1, 4)
    set_model("reference", 4, 1, 8)
    set_model("ppo_policy", 8, 1, 8)
    set_model("ppo_value", 8, 1, 8)
    chatlearn.get_args().runtime_args.colocation = [["policy", "reference", "reward", "value", "ppo_policy", "ppo_value"]]

    engine = _create_engine()
    for name in ['policy', 'reference', 'value', 'reward', 'ppo_policy', 'ppo_value']:
        setattr(engine, name, engine.named_models[name])

    for replica_id in range(len(engine.ppo_policy.replicas)):
        visible_devices = future.get(engine.ppo_policy.replicas[replica_id].get_visible_gpus())
        if replica_id == 0:
            assert visible_devices == [[0], [1], [2], [3], [4], [5], [6], [7]], visible_devices

    for replica_id in range(len(engine.policy.replicas)):
        visible_devices = future.get(engine.policy.replicas[replica_id].get_visible_gpus())
        if replica_id == 0:
            assert visible_devices == [[3], [2], [1], [0]], visible_devices
        else:
            assert visible_devices == [[7], [6], [5], [4]], visible_devices

    for replica_id in range(len(engine.reference.replicas)):
        visible_devices = future.get(engine.reference.replicas[replica_id].get_visible_gpus())
        if replica_id == 0:
            assert visible_devices == [[0], [1], [2], [3]], visible_devices
        else:
            assert visible_devices == [[4], [5], [6], [7]], visible_devices

    for replica_id in range(len(engine.value.replicas)):
        visible_devices = future.get(engine.value.replicas[replica_id].get_visible_gpus())
        assert visible_devices[0][0] == replica_id+4, visible_devices

    for replica_id in range(len(engine.reward.replicas)):
        visible_devices = future.get(engine.reward.replicas[replica_id].get_visible_gpus())
        assert visible_devices[0][0] == replica_id, visible_devices
    engine.stop()


def test_placement_colocate_3():
    set_model("policy", 4, 1, 8)
    set_model("value", 1, 1, 1)
    set_model("reward", 1, 1, 4)
    set_model("reference", 4, 1, 8)
    set_model("ppo_policy", 8, 1, 8)
    set_model("ppo_value", 8, 1, 8)
    engine = _create_engine()
    for replica_id in range(len(engine.named_models['ppo_policy'].replicas)):
        visible_devices = future.get(engine.named_models['ppo_policy'].replicas[replica_id].get_visible_gpus())
    if replica_id == 0:
        assert visible_devices == [[0], [1], [2], [3], [4], [5], [6], [7]], visible_devices

    for replica_id in range(len(engine.named_models['policy'].replicas)):
        visible_devices = future.get(engine.named_models['policy'].replicas[replica_id].get_visible_gpus())
        if replica_id == 0:
            assert visible_devices == [[3], [2], [1], [0]], visible_devices
        else:
            assert visible_devices == [[7], [6], [5], [4]], visible_devices

    for replica_id in range(len(engine.named_models['reference'].replicas)):
        visible_devices = future.get(engine.named_models['reference'].replicas[replica_id].get_visible_gpus())
        if replica_id == 0:
            assert visible_devices == [[0], [1], [2], [3]], visible_devices
        else:
            assert visible_devices == [[4], [5], [6], [7]], visible_devices

    for replica_id in range(len(engine.named_models['value'].replicas)):
        visible_devices = future.get(engine.named_models['value'].replicas[replica_id].get_visible_gpus())
        assert visible_devices[0][0] == replica_id+4, visible_devices

    for replica_id in range(len(engine.named_models['reward'].replicas)):
        visible_devices = future.get(engine.named_models['reward'].replicas[replica_id].get_visible_gpus())
        assert visible_devices[0][0] == replica_id, visible_devices
    engine.stop()

def test_placement_colocate_4():
    set_model("policy", 4, 1, 4)
    set_model("value", 1, 1, 2)
    set_model("reward", 1, 1, 4)
    set_model("reference", 2, 1, 2)
    set_model("ppo_policy", 4, 1, 4)
    set_model("ppo_value", 4, 1, 4)

    chatlearn.get_args().runtime_args.colocation = [["policy", "ppo_policy", "ppo_value"], ["reference", "reward", "value"]]
    engine = _create_engine()

    for executor in engine._executors:
        if executor:
            executor.setup()
    def check_output_models(model_name, expected_models):
        model_node = [node for node in engine.env.model_flow.model_nodes if node.name == model_name][0]
        assert [node.name for node in model_node.output_nodes] == expected_models

    check_output_models("policy", ['reference', 'value', 'reward'])
    check_output_models("reference", ['reward'])
    check_output_models("value", ['reward'])
    check_output_models("reward", [])

    def check_colocate_models(model_name, expected_models):
        model_node = [node for node in engine.env.model_flow.model_nodes if node.name == model_name][0]
        assert [model.name for model in model_node.model.colocate_models] == expected_models

    check_colocate_models("policy", ['ppo_policy', 'ppo_value'])
    check_colocate_models("reference", ['reward'])
    check_colocate_models("value", ['reward'])
    check_colocate_models("reward", ['reference', 'value'])

    def check_next_colocate_model(model_name, expected_model):
        model_node = [node for node in engine.env.model_flow.model_nodes if node.name == model_name][0]
        if model_node.next_colocate_node:
            assert model_node.next_colocate_node.name == expected_model
        else:
            assert model_node.next_colocate_node is expected_model

    check_next_colocate_model("policy", None)
    check_next_colocate_model("reference", "reward")
    check_next_colocate_model("value", "reward")
    check_next_colocate_model("reward", None)

    engine.stop()

TEST_CASE = [test_placement_colocate_2, test_placement_colocate_3, test_placement_colocate_4]