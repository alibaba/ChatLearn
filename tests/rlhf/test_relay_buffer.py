import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import ray

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
        query = data["policy_out"].cuda()
        data["ref_out"] = query * 2
        return data


class RewardModel(TorchModule):

    def forward_step(self, data, iteration):
        data["reward_out"] = data["ref_out"].cuda() + data["policy_out"].cuda()
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


def test_replay_buffer():
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    chatlearn.get_args().runtime_args.max_replay_episode = 5
    chatlearn.get_args().runtime_args.num_episode = 3
    chatlearn.get_args().runtime_args.sample_per_episode = 16
    chatlearn.get_args().runtime_args.stream_data_loader_type = "replay"

    sample_per_episode = chatlearn.get_args().runtime_args.sample_per_episode

    class ReplaySampleManagerTester(ReplaySampleManager):
            def __call__(self, episode_replay_buffers):
                buffer = episode_replay_buffers[-1].buffer
                episode_id = episode_replay_buffers[-1]._episode_id
                assert len(buffer) == 1024
                for i in range(len(buffer)):
                    assert int(buffer[i]['query'][0].item()) == i + episode_id * 1024
                return buffer


    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
    replay_sample_manager = ReplaySampleManagerTester(chatlearn.get_args())
    engine.set_replay_sample_manager(replay_sample_manager)
    assert policy.num_replica == 1
    assert reference.num_replica == 1
    data = []
    for i in range(35):
        data.append(torch.ones([10])*i)
    engine.set_dataset(data)
    engine.learn()
    assert len(engine.env._all_datasets[0]) == 35, len(engine.env._all_datasets[0])
    ref = engine._data_loader.episode_replay_buffers.remote()
    episode_replay_buffers = ray.get(ref)
    micro_batch_per_episode = ray.get(engine._data_loader.batch_per_episode.remote())
    assert micro_batch_per_episode == 10, micro_batch_per_episode
    assert engine.env.batch_per_episode == 4
    assert engine.trainer.num_iteration() == 5

    engine.stop()

TEST_CASE = [test_replay_buffer]
#TODO breaked from some reason, need to be fixed
TEST_CASE = []