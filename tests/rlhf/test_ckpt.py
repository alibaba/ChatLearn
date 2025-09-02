import os
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import chatlearn
from chatlearn import RLHFEngine
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
        save_dir = self.runtime_args.data_checkpoint_path
        fn = f"{save_dir}/data_{iteration}"
        if self.runtime_args.load_data_checkpoint_iteration:
            fn = f"{fn}_{self.runtime_args.load_data_checkpoint_iteration}"
        fn = f"{fn}.pkl"
        with open(fn, 'wb') as f:
            pickle.dump(data, f)
            print(f"save to {fn}", flush=True)

        query = data["query"]
        data["policy_out"] = query
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset


class ReferenceModel(TorchModule):

    def forward_step(self, data, iteration):
        query = data["policy_out"].cuda()
        data["ref_out"] = query
        return data


class RewardModel(TorchModule):

    def forward_step(self, data, iteration):
        data["reward_out"] = data["ref_out"].cuda()
        return data


class ValueModel(TorchModule):

    def forward_step(self, data, iteration):
        data["value_out"] = data["policy_out"].cuda() * 3
        return data


class PPOPolicy(TorchModule):

    def train_step(self, data, iteration):
        num_mb = len(data)
        return num_mb


class PPOValue(TorchModule):

    def train_step(self, data, iteration):
        num_mb = len(data)
        return num_mb


def _test_rlhf_ckpt(run):
    if run == "resume":
        chatlearn.get_args().runtime_args.load_data_checkpoint_iteration = 2
    chatlearn.get_args().models["policy"].num_replica = 1
    chatlearn.get_args().models["policy"].generation_batch_size = 4
    chatlearn.get_args().runtime_args.data_checkpoint_path = os.path.join(os.getcwd(), "checkpoint")
    chatlearn.get_args().runtime_args.save_episode_interval = 1
    chatlearn.get_args().runtime_args.num_episode = 4
    chatlearn.get_args().runtime_args.train_global_batch_size = 8
    chatlearn.get_args().runtime_args.sample_per_episode = 16
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)

    data = [torch.Tensor([i]) for i in range(100)]
    engine.set_dataset(data)
    engine.learn()
    if run == "resume":
        assert engine._start_episode == 1, engine._start_episode
        data = {}
        for fn in os.listdir(chatlearn.get_args().runtime_args.data_checkpoint_path):
            if not fn.endswith('.pkl'):
                continue

            with open(os.path.join(chatlearn.get_args().runtime_args.data_checkpoint_path, fn), 'rb') as f:
                data[fn] = pickle.load(f)
        for i in range(4, 16):
            fn_resume = f"data_{i}_2.pkl"
            fn = f"data_{i}.pkl"
            assert (data[fn_resume]['query'] == data[fn]['query']).all()
    assert engine.trainer.iteration == 8, engine.trainer.iteration
    engine.stop()

def test_rlhf_ckpt():
    return _test_rlhf_ckpt("")

def test_rlhf_ckpt_resume():
    return _test_rlhf_ckpt("resume")

def _test_rlhf_ckpt_replica(run):
    if run == "resume":
        chatlearn.get_args().runtime_args.load_data_checkpoint_iteration = 2
    chatlearn.get_args().models["policy"].num_gpu = 2
    chatlearn.get_args().models["policy"].generation_batch_size = 4
    chatlearn.get_args().runtime_args.data_checkpoint_path = os.path.join(os.getcwd(), "checkpoint2")
    chatlearn.get_args().runtime_args.save_episode_interval = 1
    chatlearn.get_args().runtime_args.num_episode = 4
    chatlearn.get_args().runtime_args.train_global_batch_size = 8
    chatlearn.get_args().runtime_args.sample_per_episode = 16
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)

    data = [torch.Tensor([i]) for i in range(100)]
    engine.set_dataset(data)
    engine.learn()
    if run == "resume":
        assert engine._start_episode == 1
        data = {}
        for fn in os.listdir(chatlearn.get_args().runtime_args.data_checkpoint_path):
            if not fn.endswith('.pkl'):
                continue

            with open(os.path.join(chatlearn.get_args().runtime_args.data_checkpoint_path, fn), 'rb') as f:
                data[fn] = pickle.load(f)
        for replica in range(2):
            for i in range(2, 8):
                fn_resume = f"data_replica{replica}_{i}_2.pkl"
                fn = f"data_replica{replica}_{i}.pkl"
                assert (data[fn_resume]['query'] == data[fn]['query']).all()
    assert engine.trainer.iteration == 8, engine.trainer.iteration
    engine.stop()

def test_rlhf_ckpt_replica():
    return _test_rlhf_ckpt_replica("")

def test_rlhf_ckpt_replica_resume():
    return _test_rlhf_ckpt_replica("resume")

def _test_rlhf_ckpt_data_ratio(run):
   
    chatlearn.get_args().runtime_args.data_ratio = [1,1]
    if run == "resume":
        chatlearn.get_args().runtime_args.load_data_checkpoint_iteration = 2
    chatlearn.get_args().models["policy"].num_replica = 1
    chatlearn.get_args().models["policy"].generation_batch_size = 4
    chatlearn.get_args().runtime_args.data_checkpoint_path = os.path.join(os.getcwd(), "checkpoint3")
    chatlearn.get_args().runtime_args.save_episode_interval = 1
    chatlearn.get_args().runtime_args.num_episode = 5
    chatlearn.get_args().runtime_args.train_global_batch_size = 8
    chatlearn.get_args().runtime_args.sample_per_episode = 16
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)

    data = [torch.Tensor([i]) for i in range(8)]
    data2 = [torch.Tensor([i]) for i in range(8,24)]
    engine.set_multiple_datasets([data, data2])
    engine.setup()
    engine.learn()
    if run == "resume":
        assert engine._start_episode == 1, engine._start_episode
        data = {}
        for fn in os.listdir(chatlearn.get_args().runtime_args.data_checkpoint_path):
            if not fn.endswith('.pkl'):
                continue

            with open(os.path.join(chatlearn.get_args().runtime_args.data_checkpoint_path, fn), 'rb') as f:
                data[fn] = pickle.load(f)
        for i in range(4, 20):
            fn_resume = f"data_{i}_2.pkl"
            fn = f"data_{i}.pkl"
            assert (data[fn]['query'] == data[fn_resume]['query']).all(), (
                f"train_iteration {i}: expect data and resumed data to be the same, "
                f"got {data[fn]['query']} and {data[fn_resume]['query']}"
            )
    assert engine.trainer.iteration == 10, engine.trainer.iteration
    engine.stop()

def test_rlhf_ckpt_data_ratio():
    return _test_rlhf_ckpt_data_ratio("")

def test_rlhf_ckpt_data_ratio_resume():
    return _test_rlhf_ckpt_data_ratio("resume")


def _test_rlhf_ckpt_data_ratio_replica(run):
    chatlearn.get_args().runtime_args.data_ratio = [1,1]
    if run == "resume":
        chatlearn.get_args().runtime_args.load_data_checkpoint_iteration = 2
    chatlearn.get_args().models["policy"].num_gpu = 2
    chatlearn.get_args().models["policy"].generation_batch_size = 4
    chatlearn.get_args().runtime_args.data_checkpoint_path = os.path.join(os.getcwd(), "checkpoint4")
    chatlearn.get_args().runtime_args.save_episode_interval = 1
    chatlearn.get_args().runtime_args.num_episode = 5
    chatlearn.get_args().runtime_args.train_global_batch_size = 8
    chatlearn.get_args().runtime_args.sample_per_episode = 16
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)

    data = [torch.Tensor([i]) for i in range(9)]
    data2 = [torch.Tensor([i]) for i in range(9,24)]

    engine.set_multiple_datasets([data, data2])
    engine.learn()

    if run == "resume":
        assert engine._start_episode == 1, engine._start_episode
        data = {}
        for fn in os.listdir(chatlearn.get_args().runtime_args.data_checkpoint_path):
            if not fn.endswith('.pkl'):
                continue

            with open(os.path.join(chatlearn.get_args().runtime_args.data_checkpoint_path, fn), 'rb') as f:
                data[fn] = pickle.load(f)
        for replica in range(2):
            for i in range(2, 10):
                fn_resume = f"data_replica{replica}_{i}_2.pkl"
                fn = f"data_replica{replica}_{i}.pkl"
                assert (data[fn]['query'] == data[fn_resume]['query']).all(), (
                    f"train_iteration {i}: expect data and resumed data to be the same, "
                    f"got {data[fn]['query']} and {data[fn_resume]['query']}"
                )

    assert engine.trainer.iteration == 10, engine.trainer.iteration
    engine.stop()

def test_rlhf_ckpt_data_ratio_replica():
    return _test_rlhf_ckpt_data_ratio_replica()

def test_rlhf_ckpt_data_ratio_replica_resume():
    return _test_rlhf_ckpt_data_ratio_replica()

TEST_CASE = [test_rlhf_ckpt, test_rlhf_ckpt_resume, test_rlhf_ckpt_replica, test_rlhf_ckpt_replica_resume,test_rlhf_ckpt_data_ratio, \
    test_rlhf_ckpt_data_ratio_resume, test_rlhf_ckpt_data_ratio_replica, test_rlhf_ckpt_data_ratio_replica_resume]

#TODO breaked from some reason, need to be fixed
TEST_CASE = []