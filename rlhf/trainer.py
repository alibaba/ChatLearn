import math
import ray
from rlhf.logger import logger


class BaseTrainer:
    """
    base trainer
    """
    def __init__(self, args):
        self.args = args


class PPOTrainer(BaseTrainer):
    """
    PPO Trainer
    """
    def __init__(self, args, ppo_policy_model, ppo_value_model):
        super().__init__(args)
        self.ppo_policy_model = ppo_policy_model
        self.ppo_value_model = ppo_value_model
        self.models = [ppo_policy_model, ppo_value_model]
        self.num_training_iteration = math.ceil(args.sample_per_episode / args.train_global_batch_size)
        self.num_micro_batch = args.train_global_batch_size // args.train_micro_batch_size
        self.iteration = 0


    def setup(self):
        pass

    
    def train_step(self, train_data, train_info):
        value_loss = self.ppo_value_model.train_step(train_data, train_info)
        policy_loss = self.ppo_policy_model.train_step(train_data, train_info)
        ray.get(value_loss + policy_loss)
        return 'ok'


    def set_data_loader(self, data_loader):
        self._data_loader = data_loader


    def next_batch(self, iteration):
        batches = []
        for _ in range(self.num_micro_batch):
            data = self._data_loader.next.remote()
            if ray.get(self._data_loader.has_next.remote()):
                batches.append(data)
        if not batches:
            return
        else:
            if len(batches) < self.num_micro_batch:
                batches += batches[:self.num_micro_batch-len(batches)]
            return batches

    
    def train(self):
        for epoch in range(self.args.num_training_epoch):
            if epoch > 0:
                ret = self._data_loader.shuffle.remote()
                ray.get(ret)
            for step in range(self.num_training_iteration):
                train_data = self.next_batch(self.iteration)
                if train_data:
                    train_info = {"iteration": self.iteration}
                    self.train_step(train_data, train_info)
                    self.iteration += 1
                    logger.info(f"train epoch {epoch} step {step} iteration {self.iteration}")
                if self.args.save_interval is not None and self.iteration % self.args.save_interval == 0:
                    ref0 = self.ppo_policy_model.save_checkpoint(self.iteration)
                    ref1 = self.ppo_value_model.save_checkpoint(self.iteration)
                    ray.get(ref0+ref1)
