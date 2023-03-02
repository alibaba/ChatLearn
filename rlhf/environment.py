import math
import random
import ray
import torch
from ray.util.queue import Queue
from rlhf.data import StreamDataset, RLHFDataLoader


class BaseEnv:
    def __init__(self, args):
        self.args = args


class PPOEnv(BaseEnv):
    """
    PPO environment
    """
    def __init__(self, args, policy, reference, reward, value):
        super().__init__(args)
        self.sample_per_episode = args.sample_per_episode
        self.policy = policy
        self.reference = reference
        self.reward = reward
        self.value = value
        self.num_rollout_worker = args.num_rollout_worker
        self.remote_models = [policy, reference, reward, value]
        self.batch_size = args.generation_batch_size
        assert args.sample_per_episode % self.batch_size == 0, "currently sample_per_episode should be times of generation_batch_size"
        self.batch_per_episode = math.ceil(args.sample_per_episode / self.batch_size)
        self._dataset = None


    def setup(self):
        data_loader = self.build_data_loader()
        self.data_iter = iter(data_loader)


    def set_dataset(self, dataset):
        self._dataset = dataset


    def build_data_loader(self):
        """generate prompts data loader"""
        return RLHFDataLoader(self._dataset, self.batch_size)


    def update_weight(self):
        """
        update model weight
        """
        pass

    def recv_weight(self):
        """
        receive latest weights
        """
        pass


    def preprocess_weight(self):
        """
        merge weight before update
        """
        pass

    def generate_step(self, query):
        # TODO: current only supports one replica, so get the first index of value
        policy_output = self.policy.forward_step(query)
        assert len(policy_output) == 1, "current only supports one replica, so get the first index of value"
        ref_output = self.reference.forward_step(policy_output[0])
        old_values = self.value.forward_step(policy_output[0])
        reward_output = self.reward.forward_step(policy_output[0], ref_output[0], old_values[0])
        return policy_output[0], ref_output[0], reward_output[0]


    def make_experiences(self):
        """
        Generate a collection of experiences for one episode
        """
        queue = Queue()
        # TODO: support num_rollout_worker > 0
        for i in range(self.batch_per_episode):
            query = next(self.data_iter)
            data = self.generate_step(query)
            queue.put(data)
        return StreamDataset.remote(queue, self.sample_per_episode, self.args.train_batch_size, cache=True)
