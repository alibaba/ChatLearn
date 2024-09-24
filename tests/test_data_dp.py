import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import chatlearn
from chatlearn import RLHFEngine
from chatlearn import TorchModule
from chatlearn.utils import future


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
        bs = query.size(0)
        data["policy_out"] = torch.ones([bs, 1024]).cuda()
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []

    def _get_rank(self):
        return int(os.environ["RANK"])

    @property
    def data_parallel_size(self):
        return 2

    @property
    def data_parallel_rank(self):
        if self._get_rank() < 4:
            return 0
        return 1

    def train_step(self, data, iteration):
        print(f"ppo policy train_step ========= {self.data_parallel_rank}", flush=True)
        if self._get_rank() == 0 or self._get_rank() == 4:
            self.data.append(data)
        num_mb = len(data)
        return num_mb

    def get_data(self):
        return self.data


class PPOValue(TorchModule):

    @property
    def data_parallel_size(self):
        return 2

    @property
    def data_parallel_rank(self):
        if int(os.environ["RANK"]) < 4:
            return 0
        return 1

    def train_step(self, data, iteration):
        print("ppo value train_step =========", flush=True)
        num_mb = len(data)
        return num_mb


chatlearn.init()
for _, model_config in chatlearn.get_args().models.items():
    model_config.num_gpu = 8

chatlearn.get_args().models['policy'].tensor_model_parallel_size = 2
chatlearn.get_args().models['reference'].tensor_model_parallel_size = 4
chatlearn.get_args().models['reward'].pipeline_model_parallel_size = 4
chatlearn.get_args().models['value'].pipeline_model_parallel_size = 4

chatlearn.get_args().models['ppo_policy'].pipeline_model_parallel_size = 2
chatlearn.get_args().models['ppo_value'].pipeline_model_parallel_size = 2
chatlearn.get_args().models['ppo_policy'].tensor_model_parallel_size = 2
chatlearn.get_args().models['ppo_value'].tensor_model_parallel_size = 2

chatlearn.get_args().runtime_args.colocation = [["policy", "reference", "reward", "value", "ppo_policy", "ppo_value"]]
chatlearn.get_args().runtime_args.train_micro_batch_size = 4
chatlearn.get_args().runtime_args.train_global_batch_size = 16
chatlearn.get_args().runtime_args.generation_batch_size = 8
chatlearn.get_args().runtime_args.max_relay_episode = 1
chatlearn.get_args().runtime_args.sample_per_episode = 256
policy = PolicyModel("policy")
reference = ReferenceModel("reference")
reward = RewardModel("reward")
value = ValueModel("value")
ppo_policy = PPOPolicy("ppo_policy")
ppo_value = PPOValue("ppo_value")

engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
def relay_sample_fn(episode_relay_buffers):
    buffer = episode_relay_buffers[-1].buffer
    episode_id = episode_relay_buffers[-1]._episode_id
    assert len(buffer) == 256
    for i in range(len(buffer)):
        assert int(buffer[i]['query'][0].item()) == i + episode_id * 256
    return buffer

engine.set_relay_sample_fn(relay_sample_fn)
assert policy.num_replica == 4
assert reference.num_replica == 2
assert reward.num_replica == 2
assert value.num_replica == 2
assert ppo_policy.num_replica == 1
assert ppo_value.num_replica == 1
data = [torch.ones([1024]) * i for i in range(512)]
engine.set_dataset(data)
engine.learn()
assert engine.named_models['policy'].replicas[0].data_parallel_size == 1
assert engine.named_models['reference'].replicas[0].data_parallel_size == 1
assert engine.named_models['reward'].replicas[0].data_parallel_size == 1
assert engine.named_models['value'].replicas[0].data_parallel_size == 1
assert engine.named_models['ppo_policy'].replicas[0].data_parallel_size == 2
assert engine.named_models['ppo_value'].replicas[0].data_parallel_size == 2

dp_rank_to_actors = engine.named_models['ppo_policy'].replicas[0].dp_rank_to_actors
assert len(dp_rank_to_actors) == 2
assert len(dp_rank_to_actors[0]) == 4
assert len(dp_rank_to_actors[1]) == 4

data0 = future.get(dp_rank_to_actors[0][0].get_data.remote())
data1 = future.get(dp_rank_to_actors[1][0].get_data.remote())

all_data = []
for item in data0+data1:
    for batch in item:
        all_data.extend([i for i in batch['query'][:, 0].numpy()])

assert len(all_data) == 512
distinct_data = set(all_data)
assert len(distinct_data) == 512
assert min(distinct_data) == 0.0
assert max(distinct_data) == 511.0

dp_rank_to_actors = engine.named_models['ppo_value'].replicas[0].dp_rank_to_actors
assert len(dp_rank_to_actors) == 2
assert len(dp_rank_to_actors[0]) == 4
assert len(dp_rank_to_actors[1]) == 4

assert engine.env.batch_per_episode == 64
assert engine.env.num_iteration == 64
assert engine.trainer.batch_per_episode == 16
assert engine.trainer.num_iteration == 16
assert engine.trainer.num_micro_batch_per_dp == 2

assert len(engine.env._dataset) == 512, len(engine.env._dataset)
