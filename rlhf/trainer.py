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
        self.num_training_iteration = math.ceil(args.sample_per_episode / args.train_batch_size)

    def setup(self):
        pass

    
    def send_weight(self):
        """
        send weight from learner to rollout_workers
        """
        pass

    
    def save_checkpoints(self):
        pass

    
    def train_step(self, train_data):
        value_loss = self.ppo_value_model.train_step(train_data)
        _, curr_log_probs = self.ppo_policy_model.train_step(train_data)
        return value_loss


    def train(self, data_loader):
        for epoch in range(self.args.num_training_epoch):
            if epoch > 0:
                ret = data_loader.shuffle.remote()
                ray.get(ret)
            for step in range(self.num_training_iteration):
                train_data = data_loader.next.remote()
                self.train_step(train_data)
        self.update_model_weight(self.ppo_policy_model)
        self.update_model_weight(self.ppo_value_model)


    def update_model_weight(self, model):
        status = model.send_weight()
        return status

