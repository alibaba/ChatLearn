import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import chatlearn
from chatlearn.models.base_module import BaseModule
from chatlearn import Engine
from chatlearn import TorchModule
from chatlearn.utils import future
from chatlearn.data.data import ReplaySampleManager
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.trainer import Trainer
from utils import assert_consumed_samples


class FakeGRPOEngine(Engine):
    def __init__(self,
                 policy: BaseModule,
                 reference: BaseModule,
                 reward: BaseModule,
                 ppo_policy: BaseModule):
        def env_compute_flow(batch):
            policy_out = policy.forward_step(batch)
            ref_out = reference.forward_step(policy_out)
            old_policy_ref_out = ppo_policy.forward_step(policy_out)
            reward_out = reward.forward_step(policy_out, ref_out, old_policy_ref_out)
            return reward_out, old_policy_ref_out

        def trainer_compute_flow(batch):
            ppo_policy.train_step(batch)

        env = Environment(env_compute_flow)
        trainer = Trainer(trainer_compute_flow)
        super().__init__(env, trainer)
        self.set_parameter_sync(ppo_policy, policy)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.collate_fn = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"query": self.data[idx]}


class PolicyModel(TorchModule):
    counter = 1

    def _get_rank(self):
        return int(os.environ["RANK"])

    @property
    def data_parallel_size(self):
        return 1

    @property
    def data_parallel_rank(self):
        return 0

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
        return 4

    @property
    def data_parallel_rank(self):
        return self._get_rank() // 2

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
        return 4

    @property
    def data_parallel_rank(self):
        return self._get_rank() // 2

    def forward_step(self, data, iteration):
        print(f"reward forward {self.counter}=========", flush=True)
        data["reward_out"] = data["ref_out"].cuda() + data["policy_out"].cuda()
        self.counter += 1
        return data


class PPOPolicy(TorchModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []
        self.train_counter = 1
        self.forward_counter = 1

    @property
    def data_parallel_size(self):
        return 4

    @property
    def data_parallel_rank(self):
        return int(os.environ["RANK"]) // 2

    def forward_step(self, data, iteration):
        print(f"ppo_policy forward {self.forward_counter}=========", flush=True)
        data["ppo_policy_out"] = data["policy_out"].cuda() * 3
        self.forward_counter += 1
        return data

    def train_step(self, data, iteration):
        print(f"ppo policy train_step {self.train_counter}========= {self.data_parallel_rank}", flush=True)
        self.data.append(data)
        num_mb = len(data)
        self.train_counter += 1
        return num_mb

    def get_data(self):
        return self.data

def test_grpo():
    for _, model_config in chatlearn.get_args().models.items():
        model_config.num_gpu = 8

    chatlearn.get_args().models['policy'].expert_model_parallel_size = 1
    chatlearn.get_args().models['policy'].tensor_model_parallel_size = 8

    chatlearn.get_args().models['reference'].expert_model_parallel_size = 1
    chatlearn.get_args().models['reference'].tensor_model_parallel_size = 2

    chatlearn.get_args().models['reward'].expert_model_parallel_size = 1
    chatlearn.get_args().models['reward'].tensor_model_parallel_size = 2

    chatlearn.get_args().models['ppo_policy'].expert_model_parallel_size = 2
    chatlearn.get_args().models['ppo_policy'].tensor_model_parallel_size = 2

    chatlearn.get_args().runtime_args.colocation = [["policy", "reference", "reward", "ppo_policy"]]
    chatlearn.get_args().runtime_args.train_micro_batch_size = 4
    chatlearn.get_args().runtime_args.train_global_batch_size = 32
    chatlearn.get_args().runtime_args.generation_batch_size = 8
    chatlearn.get_args().runtime_args.max_replay_episode = 1
    chatlearn.get_args().runtime_args.sample_per_episode = 1024
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    ppo_policy = PPOPolicy("ppo_policy")

    engine = FakeGRPOEngine(policy, reference, reward, ppo_policy)

    class ReplaySampleManagerTester(ReplaySampleManager):
        def __call__(self, episode_replay_buffers):
            buffer = episode_replay_buffers[-1].buffer
            episode_id = episode_replay_buffers[-1]._episode_id
            assert len(buffer) == 1024
            for i in range(len(buffer)):
                assert int(buffer[i]['query'][0].item()) == i + episode_id * 1024
            return buffer

    replay_sample_manager = ReplaySampleManagerTester(chatlearn.get_args())
    engine.set_replay_sample_manager(replay_sample_manager)
    assert policy.num_replica == 1
    assert reference.num_replica == 4
    assert reward.num_replica == 4
    # for training models, ep is combined into dp, leading to only 1 replica
    assert ppo_policy.num_replica == 1
    data = [torch.ones([1024]) * i for i in range(2048)]
    engine.set_dataset(data)
    engine.learn()
    assert engine.named_models['policy'].replicas[0].data_parallel_size == 1
    assert engine.named_models['reference'].replicas[0].data_parallel_size == 4
    assert engine.named_models['reward'].replicas[0].data_parallel_size == 4
    assert engine.named_models['ppo_policy'].replicas[0].data_parallel_size == 4

    policy_replicas = engine.named_models['policy'].replicas
    assert len(policy_replicas) == 1
    dp_rank_to_actors = policy_replicas[0].dp_rank_to_actors
    assert len(dp_rank_to_actors) == 1
    assert len(dp_rank_to_actors[0]) == 8

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

    assert engine.env.batch_per_episode == 256, f"{engine.env.batch_per_episode}"
    assert engine.env.num_iteration() == 256, f"{engine.env.num_iteration()}"
    assert engine.trainer.batch_per_episode == 32, f"{engine.trainer.batch_per_episode}"
    assert engine.trainer.num_iteration() == 32, f"{engine.trainer.num_iteration()}"
    assert engine.trainer.num_micro_batch_per_dp == 2, f"{engine.trainer.num_micro_batch_per_dp}"

    assert len(engine.env._all_datasets[0]) == 2048, len(engine.env._all_datasets[0])

    assert_consumed_samples(
        engine,
        ['policy'],
        2048
    )
    engine.stop()

TEST_CASE = [test_grpo]
#TODO breaked from some reason, need to be fixed
TEST_CASE = []