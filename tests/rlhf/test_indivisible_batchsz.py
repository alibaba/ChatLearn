import torch
from torch.utils.data import DataLoader, Dataset
import ray
import random

import chatlearn
from chatlearn import RLHFEngine
from chatlearn import TorchModule
from chatlearn.utils import future
from chatlearn.data.data import StreamDataset


class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"query": self.data[idx]}

    def collate_fn(self, samples):
        batched_data = {}
        for sample in samples:
            for sample_key, sample_value in sample.items():
                if sample_key not in batched_data:
                    batched_data[sample_key] = []
                batched_data[sample_key].append(sample_value)
        for sample_key, sample_value in batched_data.items():
            batched_data[sample_key] = torch.stack(sample_value)
        return batched_data


class PolicyModel(TorchModule):

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        data["policy_out"] = data["query"].cuda()
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset


class ReferenceModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reference forward =========", flush=True)
        query = data["policy_out"].cuda()
        data["ref_out"] = query * 2
        return data


class RewardModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reward forward =========", flush=True)
        data["reward_out"] = data["ref_out"].cuda() + data["policy_out"].cuda()
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


def get_batches(modify_generation_batch_size=False):
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    if modify_generation_batch_size:
        new_generation_batch_size = 5
        policy.module_args.generation_batch_size = new_generation_batch_size
        reference.module_args.generation_batch_size = new_generation_batch_size
        reward.module_args.generation_batch_size = new_generation_batch_size
        value.module_args.generation_batch_size = new_generation_batch_size

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
    assert policy.num_replica == 1
    assert reference.num_replica == 1
    data = torch.ones([1024])
    dataset = [data * i for i in range(35)]
    engine.set_dataset(dataset)
    engine.setup()
    engine.trainer.setup()
    engine.env.setup()
    if engine.evaluator:
        engine.evaluator.setup()

    data_loader = StreamDataset.remote(engine.runtime_args.stream_data_loader_type,
                                            engine.runtime_args.train_micro_batch_size,
                                            engine.runtime_args.max_replay_episode,
                                            engine.runtime_args.replay_episode_offset)
    engine._data_loader = data_loader
    engine.trainer.num_micro_batch_per_dp = engine.trainer.args.train_global_batch_size // \
        engine.trainer.args.train_micro_batch_size // engine.trainer.data_parallel_size
    train_datas = []
    args = chatlearn.get_args()
    sample_per_episode = chatlearn.get_args().runtime_args.sample_per_episode

    for episode_id in range(5):
        print(f"Testing episode {episode_id}...")
        engine.before_episode()
        queue = engine.env.make_experiences()
        refs = data_loader.set_dataset.remote(queue, episode_id, sample_per_episode=sample_per_episode)
        future.wait(refs)
        engine.trainer.set_data_loader(data_loader)
        train_datas.extend([engine.trainer.next_batch() for step in range(2)])
        engine.evaluate(episode_id)
    engine.stop()
    return train_datas

def test_indivisible_batch_size():
   
    divisible_batches = get_batches()
    indivisible_batches = get_batches(modify_generation_batch_size=True)

    assert len(divisible_batches) == len(indivisible_batches), \
        f"Divisible data loader has {len(divisible_batches)} batches, " \
        f"while indivisible data loader has {len(indivisible_batches)} batches."

    for divisible_data_refs, indivisible_data_refs in list(zip(divisible_batches, indivisible_batches)):
        if divisible_data_refs is None and indivisible_data_refs is None:
            continue
        for divisible_data_ref, indivisible_data_ref in list(zip(divisible_data_refs, indivisible_data_refs)):
            divisible_data = future.get(divisible_data_ref)
            indivisible_data = future.get(indivisible_data_ref)
            assert divisible_data.keys() == indivisible_data.keys(), \
                f"divisible data has different keys with indivisible data"
            for key in divisible_data.keys():
                assert torch.equal(divisible_data[key], indivisible_data[key]), \
                    f"On key {key}, divisible generation batch size must have the same result data with indivisible one. "\
                    f"However, divisible: {divisible_data[key]}, indivisible:{indivisible_data[key]}"

    print("test_indivisible_batchsz passed!")

TEST_CASE = [test_indivisible_batch_size]
#TODO breaked from some reason, need to be fixed
TEST_CASE = []