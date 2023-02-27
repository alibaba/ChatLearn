import math
import random
from ray.util.queue import Queue
from torch.utils.data import IterableDataset


class StreamDataset(IterableDataset):
    """dataset built from queues"""

    def __init__(self, queue, num_batches, cache=False):
        super(StreamDataset).__init__()
        self.queue = queue
        self.num_batches = num_batches
        self.produce_index = 0
        self.cache = cache
        self.relay_buffer = []


    def shuffle(self):
        random.shuffle(self.relay_buffer)


    def __iter__(self):
        self.produce_index = 0
        while self.produce_index < self.num_batches:
            # read from cache
            if len(self.relay_buffer) == self.num_batches:
                data = self.cache_data[self.produce_index]
            else:
                # get from queue
                data = self.queue.get()
            if self.cache:
                self.relay_buffer.append(data)
            yield data
            self.produce_index += 1


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
        self.batch_per_episode = math.ceil(args.sample_per_episode / self.batch_size)

    def setup(self):
        for remote_model in self.remote_models:
            remote_model.setup()

    def build_data_loader(self):
        """generate prompts data loader"""
        # TODO: temply
        return iter([i for i in range(100000)])

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
        policy_output = self.policy.forward_step(query)
        ref_output = self.reference.forward_step(policy_output)
        old_values = self.value.forward_step(policy_output)
        reward_output = self.reward.forward_step([policy_output, ref_output, old_values])
        return [policy_output, ref_output, reward_output]


    def make_experiences(self):
        """
        Generate a collection of experiences for one episode
        """
        data_loader = self.build_data_loader()
        queue = Queue()
        # TODO: support num_rollout_worker > 0 
        for i in range(self.batch_per_episode):
            query = next(data_loader)
            # query = self.policy.next_batch(data_loader)
            data = self.generate_step(query)
            print(data, queue.size(), '=============', flush=True)
            queue.put(data)
        return StreamDataset(queue, self.batch_per_episode, cache=True)
