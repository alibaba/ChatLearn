import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import chatlearn
from chatlearn import RLHFEngine
from chatlearn import TorchModule
from chatlearn.utils import future
from utils import assert_consumed_samples


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.collate_fn = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"query": self.data[idx]}


class PolicyModel(TorchModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_parallel_size = None

    @property
    def data_parallel_size(self):
        return self._data_parallel_size

    @property
    def data_parallel_rank(self):
        return int(os.environ["RANK"])

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_parallel_size = None

    @property
    def data_parallel_size(self):
        return self._data_parallel_size

    @property
    def data_parallel_rank(self):
        return int(os.environ["RANK"])

    def forward_step(self, data, iteration):
        print("reference forward =========", flush=True)
        query = data["policy_out"].cuda()
        data["ref_out"] = query * 2
        return data


class RewardModel(TorchModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_parallel_size = None

    @property
    def data_parallel_size(self):
        return self._data_parallel_size

    @property
    def data_parallel_rank(self):
        return int(os.environ["RANK"])

    def forward_step(self, data, iteration):
        print("reward forward =========", flush=True)
        data["reward_out"] = data["ref_out"].cuda() + data["policy_out"].cuda()
        return data

class ValueModel(TorchModule):
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_parallel_size = None

    @property
    def data_parallel_size(self):
        return self._data_parallel_size

    @property
    def data_parallel_rank(self):
        return int(os.environ["RANK"])

    def forward_step(self, data, iteration):
        print("value forward =========", flush=True)
        data["value_out"] = data["policy_out"].cuda() * 3
        return data


class PPOPolicy(TorchModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_parallel_size = None
        self.data = []

    @property
    def data_parallel_size(self):
        return self._data_parallel_size

    @property
    def data_parallel_rank(self):
        return int(os.environ["RANK"])

    def train_step(self, data, iteration):
        print(f"ppo policy train_step ========= {self.data_parallel_rank}", flush=True)
        self.data.append(data)
        num_mb = len(data)
        return num_mb

    def get_data(self):
        return self.data

class PPOValue(TorchModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_parallel_size = None

    @property
    def data_parallel_size(self):
        return self._data_parallel_size

    @property
    def data_parallel_rank(self):
        return int(os.environ["RANK"])

    def train_step(self, data, iteration):
        print("ppo value train_step =========", flush=True)
        num_mb = len(data)
        return num_mb


def test_data_dp():

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
    chatlearn.get_args().runtime_args.max_replay_episode = 1
    chatlearn.get_args().runtime_args.sample_per_episode = 256
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    ppo_policy._data_parallel_size = 2
    ppo_value._data_parallel_size = 2

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
    def replay_sample_fn(episode_replay_buffers):
        buffer = episode_replay_buffers[-1].buffer
        episode_id = episode_replay_buffers[-1]._episode_id
        assert len(buffer) == 256, f"{len(buffer)}"
        for i in range(len(buffer)):
            assert int(buffer[i]['query'][0].item()) == i + episode_id * 256
        return buffer

    engine.set_replay_sample_fn(replay_sample_fn)
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
    assert engine.env.num_iteration() == 64
    assert engine.trainer.batch_per_episode == 16
    assert engine.trainer.num_iteration() == 16
    assert engine.trainer.num_micro_batch_per_dp == 2

    assert len(engine.env._all_datasets[0]) == 512, len(engine.env._all_datasets[0])

    assert_consumed_samples(
        engine,
        ['policy', 'reference', 'reward', 'value', 'ppo_policy', 'ppo_value'],
        512
    )
    engine.stop()

def test_data_dp_ep():
    for _, model_config in chatlearn.get_args().models.items():
        model_config.num_gpu = 8
        model_config.tensor_model_parallel_size = 1
        model_config.pipeline_model_parallel_size = 1

    chatlearn.get_args().models['policy'].expert_model_parallel_size = 4
    chatlearn.get_args().models['reference'].expert_model_parallel_size = 4
    chatlearn.get_args().models['reward'].expert_model_parallel_size = 4
    chatlearn.get_args().models['value'].expert_model_parallel_size = 4

    chatlearn.get_args().models['ppo_policy'].expert_model_parallel_size = 4
    chatlearn.get_args().models['ppo_value'].expert_model_parallel_size = 4

    chatlearn.get_args().runtime_args.colocation = [["policy", "reference", "reward", "value", "ppo_policy", "ppo_value"]]
    chatlearn.get_args().runtime_args.train_micro_batch_size = 4
    chatlearn.get_args().runtime_args.train_global_batch_size = 32
    chatlearn.get_args().runtime_args.generation_batch_size = 8
    chatlearn.get_args().runtime_args.max_replay_episode = 1
    chatlearn.get_args().runtime_args.sample_per_episode = 1024
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    ppo_policy._data_parallel_size = 8
    ppo_value._data_parallel_size = 8
    policy._data_parallel_size = 8
    reference._data_parallel_size = 8
    reward._data_parallel_size = 8
    value._data_parallel_size = 8

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)

    def replay_sample_fn(episode_replay_buffers):
        buffer = episode_replay_buffers[-1].buffer
        episode_id = episode_replay_buffers[-1]._episode_id
        assert len(buffer) == 1024, f"Unexpected length of buffer: {len(buffer)}, expected: 1024."
        for i in range(len(buffer)):
            assert int(buffer[i]['query'][0].item()) == i + episode_id * 1024
        return buffer

    engine.set_replay_sample_fn(replay_sample_fn)
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
    assert engine.named_models['policy'].replicas[0].data_parallel_size == 8
    assert engine.named_models['reference'].replicas[0].data_parallel_size == 8
    assert engine.named_models['reward'].replicas[0].data_parallel_size == 8
    assert engine.named_models['value'].replicas[0].data_parallel_size == 8
    assert engine.named_models['ppo_policy'].replicas[0].data_parallel_size == 8
    assert engine.named_models['ppo_value'].replicas[0].data_parallel_size == 8

    dp_rank_to_actors = engine.named_models['policy'].replicas[0].dp_rank_to_actors
    assert len(dp_rank_to_actors) == 4
    assert len(dp_rank_to_actors[0]) == 1
    assert len(dp_rank_to_actors[1]) == 1

    dp_rank_to_actors = engine.named_models['ppo_policy'].replicas[0].dp_rank_to_actors
    assert len(dp_rank_to_actors) == 8
    assert len(dp_rank_to_actors[0]) == 1
    assert len(dp_rank_to_actors[1]) == 1

    all_data = []
    for i in range(8):
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
    assert len(dp_rank_to_actors) == 8
    assert len(dp_rank_to_actors[0]) == 1
    assert len(dp_rank_to_actors[1]) == 1

    assert engine.env.batch_per_episode == 256
    assert engine.env.num_iteration() == 64
    assert engine.trainer.batch_per_episode == 32
    assert engine.trainer.num_iteration() == 32
    assert engine.trainer.num_micro_batch_per_dp == 1

    assert len(engine.env._all_datasets[0]) == 2048, len(engine.env._all_datasets[0])

    assert_consumed_samples(
        engine,
        ['policy', 'reference', 'reward', 'value', 'ppo_policy', 'ppo_value'],
        2048
    )

    engine.stop()

def test_data_tp_2_ep():
    for _, model_config in chatlearn.get_args().models.items():
        model_config.num_gpu = 8

    chatlearn.get_args().models['policy'].expert_model_parallel_size = 1
    chatlearn.get_args().models['reference'].expert_model_parallel_size = 2
    chatlearn.get_args().models['reward'].expert_model_parallel_size = 1
    chatlearn.get_args().models['value'].expert_model_parallel_size = 1

    chatlearn.get_args().models['policy'].tensor_model_parallel_size = 4
    chatlearn.get_args().models['reference'].tensor_model_parallel_size = 4
    chatlearn.get_args().models['reward'].tensor_model_parallel_size = 8
    chatlearn.get_args().models['value'].tensor_model_parallel_size = 4

    chatlearn.get_args().models['ppo_policy'].expert_model_parallel_size = 2
    chatlearn.get_args().models['ppo_value'].expert_model_parallel_size = 2

    chatlearn.get_args().models['ppo_policy'].tensor_model_parallel_size = 4
    chatlearn.get_args().models['ppo_value'].tensor_model_parallel_size = 4

    chatlearn.get_args().runtime_args.colocation = [["policy", "reference", "reward", "value", "ppo_policy", "ppo_value"]]
    chatlearn.get_args().runtime_args.train_micro_batch_size = 4
    chatlearn.get_args().runtime_args.train_global_batch_size = 32
    chatlearn.get_args().runtime_args.max_replay_episode = 1
    chatlearn.get_args().runtime_args.sample_per_episode = 1024
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)

    def replay_sample_fn(episode_replay_buffers):
        buffer = episode_replay_buffers[-1].buffer
        episode_id = episode_replay_buffers[-1]._episode_id
        assert len(buffer) == 1024
        for i in range(len(buffer)):
            assert int(buffer[i]['query'][0].item()) == i + episode_id * 1024
        return buffer

    engine.set_replay_sample_fn(replay_sample_fn)
    # for inference models, they have 2 dp replicas
    assert policy.num_replica == 2
    assert reference.num_replica == 1
    assert reward.num_replica == 1
    assert value.num_replica == 2
    # for training models, ep is combined into dp, leading to only 1 replica
    assert ppo_policy.num_replica == 1
    assert ppo_value.num_replica == 1
    data = [torch.ones([1024]) * i for i in range(2048)]
    engine.set_dataset(data)
    engine.learn()
    assert engine.named_models['policy'].replicas[0].data_parallel_size == 2
    assert engine.named_models['reference'].replicas[0].data_parallel_size == 2
    assert engine.named_models['reward'].replicas[0].data_parallel_size == 1
    assert engine.named_models['value'].replicas[0].data_parallel_size == 2
    assert engine.named_models['ppo_policy'].replicas[0].data_parallel_size == 2
    assert engine.named_models['ppo_value'].replicas[0].data_parallel_size == 2

    policy_replicas = engine.named_models['policy'].replicas
    assert len(policy_replicas) == 2
    dp_rank_to_actors = policy_replicas[0].dp_rank_to_actors
    assert len(dp_rank_to_actors) == 1
    assert len(dp_rank_to_actors[0]) == 4

    dp_rank_to_actors = engine.named_models['ppo_policy'].replicas[0].dp_rank_to_actors
    assert len(dp_rank_to_actors) == 2
    assert len(dp_rank_to_actors[0]) == 4
    assert len(dp_rank_to_actors[1]) == 4

    all_data = []
    for i in range(2):
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
    assert len(dp_rank_to_actors) == 2
    assert len(dp_rank_to_actors[0]) == 4
    assert len(dp_rank_to_actors[1]) == 4

    assert engine.env.batch_per_episode == 256
    assert engine.env.num_iteration(engine.named_models['policy']) == 256
    assert engine.env.num_iteration(engine.named_models['reference']) == 128
    assert engine.trainer.batch_per_episode == 32
    assert engine.trainer.num_iteration() == 32
    assert engine.trainer.num_micro_batch_per_dp == 4

    assert len(engine.env._all_datasets[0]) == 2048, len(engine.env._all_datasets[0])

    assert_consumed_samples(
        engine,
        ['policy', 'reference', 'reward', 'value', 'ppo_policy', 'ppo_value'],
        2048
    )

    engine.stop()

def test_data_tp_ep_pp():
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
    chatlearn.get_args().runtime_args.max_replay_episode = 1
    chatlearn.get_args().runtime_args.sample_per_episode = 1024
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)

    def replay_sample_fn(episode_replay_buffers):
        buffer = episode_replay_buffers[-1].buffer
        episode_id = episode_replay_buffers[-1]._episode_id
        assert len(buffer) == 1024
        for i in range(len(buffer)):
            assert int(buffer[i]['query'][0].item()) == i + episode_id * 1024
        return buffer

    engine.set_replay_sample_fn(replay_sample_fn)
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

    assert len(engine.env._all_datasets[0]) == 2048, len(engine.env._all_datasets[0])

    assert_consumed_samples(
        engine,
        ['policy', 'reference', 'reward', 'value', 'ppo_policy', 'ppo_value'],
        2048
    )
    engine.stop()

def test_data_tp_ep():
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

    chatlearn.get_args().models['ppo_policy'].expert_model_parallel_size = 8
    chatlearn.get_args().models['ppo_value'].expert_model_parallel_size = 8

    chatlearn.get_args().runtime_args.colocation = [["policy", "reference", "reward", "value", "ppo_policy", "ppo_value"]]
    chatlearn.get_args().runtime_args.train_micro_batch_size = 4
    chatlearn.get_args().runtime_args.train_global_batch_size = 32
    chatlearn.get_args().runtime_args.generation_batch_size = 8
    chatlearn.get_args().runtime_args.max_replay_episode = 1
    chatlearn.get_args().runtime_args.sample_per_episode = 1024
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)

    def replay_sample_fn(episode_replay_buffers):
        buffer = episode_replay_buffers[-1].buffer
        episode_id = episode_replay_buffers[-1]._episode_id
        assert len(buffer) == 1024
        for i in range(len(buffer)):
            assert int(buffer[i]['query'][0].item()) == i + episode_id * 1024
        return buffer

    engine.set_replay_sample_fn(replay_sample_fn)
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
    assert engine.named_models['ppo_policy'].replicas[0].data_parallel_size == 8
    assert engine.named_models['ppo_value'].replicas[0].data_parallel_size == 8

    policy_replicas = engine.named_models['policy'].replicas
    assert len(policy_replicas) == 2
    dp_rank_to_actors = policy_replicas[0].dp_rank_to_actors
    assert len(dp_rank_to_actors) == 1
    assert len(dp_rank_to_actors[0]) == 4

    dp_rank_to_actors = engine.named_models['ppo_policy'].replicas[0].dp_rank_to_actors
    assert len(dp_rank_to_actors) == 8
    assert len(dp_rank_to_actors[0]) == 1
    assert len(dp_rank_to_actors[1]) == 1

    all_data = []
    for i in range(8):
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
    assert len(dp_rank_to_actors) == 8
    assert len(dp_rank_to_actors[0]) == 1
    assert len(dp_rank_to_actors[1]) == 1

    assert engine.env.batch_per_episode == 256
    assert engine.env.num_iteration() == 256
    assert engine.trainer.batch_per_episode == 32
    assert engine.trainer.num_iteration() == 32
    assert engine.trainer.num_micro_batch_per_dp == 1

    assert len(engine.env._all_datasets[0]) == 2048, len(engine.env._all_datasets[0])

    assert_consumed_samples(
        engine,
        ['policy', 'reference', 'reward', 'value', 'ppo_policy', 'ppo_value'],
        2048
    )
    engine.stop()


def test_fixed_data():
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    sample_per_episode = chatlearn.get_args().runtime_args.sample_per_episode
    chatlearn.get_args().runtime_args.max_replay_episode = 1

    def replay_sample_fn(episode_replay_buffers):
        buffers = []
        for replay_buffer in episode_replay_buffers:
            buffers += replay_buffer.buffer
        episode_id = episode_replay_buffers[-1].episode_id
        assert len(buffers) == sample_per_episode, f"{len(buffers)}, {episode_id+1}, {sample_per_episode}"
        return buffers

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
    engine.set_replay_sample_fn(replay_sample_fn)
    assert policy.num_replica == 1
    assert reference.num_replica == 1
    data = torch.ones([1024])
    engine.set_dataset([data] * 35)
    engine.learn()
    assert len(engine.env._all_datasets[0]) == 35, len(engine.env._all_datasets[0])
    ref = engine._data_loader.episode_replay_buffers.remote()
    episode_replay_buffers = ray.get(ref)
    print(episode_replay_buffers)
    micro_batch_per_episode = ray.get(engine._data_loader.batch_per_episode.remote())
    assert micro_batch_per_episode == 4
    assert engine.trainer.num_iteration() == 2
    engine.stop()


def test_dynamic_data():
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    chatlearn.get_args().runtime_args.dynamic_train_samples = True
    chatlearn.get_args().runtime_args.stream_data_loader_type = "dynamic"
    sample_per_episode = chatlearn.get_args().runtime_args.sample_per_episode

    def replay_sample_fn(episode_replay_buffers):
        buffers = []
        for replay_buffer in episode_replay_buffers:
            buffers += replay_buffer.buffer
        episode_id = episode_replay_buffers[-1].episode_id
        assert len(buffers) == (episode_id+1) * sample_per_episode, f"{len(buffers)}, {episode_id+1}, {sample_per_episode}"
        return buffers

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
    engine.set_replay_sample_fn(replay_sample_fn)
    assert policy.num_replica == 1
    assert reference.num_replica == 1
    data = torch.ones([1024])
    engine.set_dataset([data] * 35)

    engine.learn()
    assert len(engine.env._all_datasets[0]) == 35, len(engine.env._all_datasets[0])
    ref = engine._data_loader.episode_replay_buffers.remote()
    episode_replay_buffers = ray.get(ref)
    print(episode_replay_buffers)
    micro_batch_per_episode = ray.get(engine._data_loader.batch_per_episode.remote())
    assert micro_batch_per_episode == 4
    assert engine.trainer.num_iteration() == 2
    engine.stop()

TEST_CASE = [test_fixed_data, test_dynamic_data, test_data_dp, test_data_dp_ep, test_data_dp_zero, test_data_tp_2_ep, test_data_tp_ep_pp, test_data_tp_ep]
#TODO breaked from some reason, need to be fixed
TEST_CASE = []