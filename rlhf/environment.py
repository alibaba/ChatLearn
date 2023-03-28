import math
import ray
from rlhf.logger import logger

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
        self.data_iter = None
        self._padding_config = {}


    def setup(self):
        data_loader = self.build_data_loader()
        if data_loader is not None:
            self.data_iter = iter(data_loader)
        else:
            ref = self.policy.master._build_dataloader.remote(self._dataset)
            ray.get(ref)
            logger.info("set dataset for policy")

        for model in [self.policy, self.reference, self.reward, self.value]:
            config = ray.get(model.master.padding_config.remote())
            self._padding_config.update(config)


    def set_dataset(self, dataset):
        self._dataset = dataset


    def build_data_loader(self):
        """generate prompts data loader"""
        pass
        # TODO: temply comment out, use dataloader from policy, consider later
        # return RLHFDataLoader(self._dataset, self.batch_size)


    def generate_step(self, query):
        policy_output = self.policy.forward_step(query)
        ref_output = self.reference.forward_step(policy_output[0])
        old_values = self.value.forward_step(policy_output[0])
        # the three inputs are merged, so the users get one dict input in their side
        reward_output = self.reward.forward_step(policy_output[0], ref_output[0], old_values[0])
        return policy_output[0], ref_output[0], reward_output[0]


    def make_experiences(self, queue):
        """
        Generate a collection of experiences for one episode
        """
        for i in range(self.batch_per_episode):
            if self.data_iter is not None:
                query = next(self.data_iter)
            else:
                query = self.policy.master.next_batch.remote()
            data = self.generate_step(query)
            queue.put(data)
