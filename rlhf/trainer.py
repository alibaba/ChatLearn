import math
import ray


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


    
    def send_weight(self):
        """
        send weight from learner to rollout_workers
        """
        pass

    
    def save_checkpoints(self):
        pass

    
    def train_step(self, train_data):
        value_loss = self.ppo_value_model.train_step(train_data)
        policy_loss = self.ppo_policy_model.train_step(train_data)
        ray.get(value_loss)
        ray.get(policy_loss)
        return 'ok'


    def set_data_loader(self, data_loader):
        self._data_loader = data_loader


    def next_batch(self):
        batches = []
        for _ in range(self.num_micro_batch):
            batches.append(self._data_loader.next.remote())
        return batches

    
    def train(self):
        for epoch in range(self.args.num_training_epoch):
            if epoch > 0:
                ret = self._data_loader.shuffle.remote()
                ray.get(ret)
            for step in range(self.num_training_iteration):
                train_data = self.next_batch()
                self.train_step(train_data)

