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


chatlearn.init()

class PolicyModel(TorchModule):
    counter = 1

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

    def forward_step(self, data, iteration):
        print(f"policy forward {self.counter}=========", flush=True)
        query = data["query"]
        bs = query.size(0)
        data["policy_out"] = torch.ones([bs, 1024]).cuda()
        self.counter += 1
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset



class ReferenceModel(TorchModule):
    counter = 1

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

    def forward_step(self, data, iteration):
        print(f"reference forward {self.counter}=========", flush=True)
        query = data["policy_out"].cuda()
        data["ref_out"] = query * 2
        self.counter += 1
        return data


class RewardModel(TorchModule):
    counter = 1

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

    def forward_step(self, data, iteration):
        print(f"reward forward {self.counter}=========", flush=True)
        data["reward_out"] = data["ref_out"].cuda() + data["policy_out"].cuda()
        self.counter += 1
        return data

class ValueModel(TorchModule):
    counter = 1

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

    def forward_step(self, data, iteration):
        print(f"value forward {self.counter}=========", flush=True)
        data["value_out"] = data["policy_out"].cuda() * 3
        self.counter += 1
        return data


class PPOPolicy(TorchModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []
        self.counter = 1

    @property
    def data_parallel_size(self):
        return 4

    @property
    def data_parallel_rank(self):
        return int(int(os.environ["RANK"]) % 4)

    def train_step(self, data, iteration):
        print(f"ppo policy train_step {self.counter}========= {self.data_parallel_rank}", flush=True)
        self.data.append(data)
        num_mb = len(data)
        self.counter += 1
        return num_mb

    def get_data(self):
        return self.data

class PPOValue(TorchModule):
    counter = 1

    @property
    def data_parallel_size(self):
        return 4

    @property
    def data_parallel_rank(self):
        return int(int(os.environ["RANK"]) % 4)

    def train_step(self, data, iteration):
        print(f"ppo value train_step {self.counter}=========", flush=True)
        num_mb = len(data)
        self.counter += 1
        return num_mb

for _, model_config in chatlearn.get_args().models.items():
    model_config.num_gpu = 8

chatlearn.get_args().models['policy'].expert_model_parallel_size = 1
chatlearn.get_args().models['reference'].expert_model_parallel_size = 1
chatlearn.get_args().models['reward'].expert_model_parallel_size = 1
chatlearn.get_args().models['value'].expert_model_parallel_size = 1

chatlearn.get_args().models['policy'].tensor_model_parallel_size = 4
chatlearn.get_args().models['reference'].tensor_model_parallel_size = 4
chatlearn.get_args().models['reward'].tensor_model_parallel_size = 4
chatlearn.get_args().models['value'].tensor_model_parallel_size = 4

chatlearn.get_args().models['ppo_policy'].expert_model_parallel_size = 4
chatlearn.get_args().models['ppo_value'].expert_model_parallel_size = 4

chatlearn.get_args().models['ppo_policy'].pipeline_model_parallel_size = 2
chatlearn.get_args().models['ppo_value'].pipeline_model_parallel_size = 2

chatlearn.get_args().runtime_args.colocation = [["policy", "reference", "reward", "value", "ppo_policy", "ppo_value"]]
chatlearn.get_args().runtime_args.train_micro_batch_size = 4
chatlearn.get_args().runtime_args.train_global_batch_size = 32
chatlearn.get_args().runtime_args.generation_batch_size = 8
chatlearn.get_args().runtime_args.max_relay_episode = 1
chatlearn.get_args().runtime_args.sample_per_episode = 1024
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
    assert len(buffer) == 1024
    for i in range(len(buffer)):
        assert int(buffer[i]['query'][0].item()) == i + episode_id * 1024
    return buffer

engine.set_relay_sample_fn(relay_sample_fn)
# for inference models, they have 2 dp replicas
assert policy.num_replica == 2
assert reference.num_replica == 2
assert reward.num_replica == 2
assert value.num_replica == 2
# for training models, ep is combined into dp, leading to only 1 replica
assert ppo_policy.num_replica == 1
assert ppo_value.num_replica == 1
data = [torch.ones([1024]) * i for i in range(2048)]
engine.set_dataset(data)
engine.learn()
assert engine.named_models['policy'].replicas[0].data_parallel_size == 2
assert engine.named_models['reference'].replicas[0].data_parallel_size == 2
assert engine.named_models['reward'].replicas[0].data_parallel_size == 2
assert engine.named_models['value'].replicas[0].data_parallel_size == 2
assert engine.named_models['ppo_policy'].replicas[0].data_parallel_size == 4
assert engine.named_models['ppo_value'].replicas[0].data_parallel_size == 4

policy_replicas = engine.named_models['policy'].replicas
assert len(policy_replicas) == 2
dp_rank_to_actors = policy_replicas[0].dp_rank_to_actors
assert len(dp_rank_to_actors) == 1
assert len(dp_rank_to_actors[0]) == 4

dp_rank_to_actors = engine.named_models['ppo_policy'].replicas[0].dp_rank_to_actors
assert len(dp_rank_to_actors) == 4
assert len(dp_rank_to_actors[0]) == 2
assert len(dp_rank_to_actors[1]) == 2

all_data = []
for i in range(4):
    data = future.get(dp_rank_to_actors[i][0].get_data.remote())
    for item in data:
        for batch in item:
            all_data.extend([i for i in batch['query'][:, 0].numpy()])

assert len(all_data) == 2048
distinct_data = set(all_data)
assert len(distinct_data) == 2048
assert min(distinct_data) == 0.0
assert max(distinct_data) == 2047.0

dp_rank_to_actors = engine.named_models['ppo_value'].replicas[0].dp_rank_to_actors
assert len(dp_rank_to_actors) == 4
assert len(dp_rank_to_actors[0]) == 2
assert len(dp_rank_to_actors[1]) == 2

assert engine.env.batch_per_episode == 256
assert engine.env.num_iteration() == 256
assert engine.trainer.batch_per_episode == 32
assert engine.trainer.num_iteration() == 32
assert engine.trainer.num_micro_batch_per_dp == 2

assert len(engine.env._dataset) == 2048, len(engine.env._dataset)
