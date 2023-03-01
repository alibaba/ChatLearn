import ray
from rlhf.model_manager import ModelManager
from rlhf.resource import ResourceManager
from rlhf.model_wrapper import RLHFModelWrapper
from rlhf.environment import PPOEnv
from rlhf.trainer import PPOTrainer
from rlhf.global_vars import get_args

class Engine:

    def __init__(self, *models):
        global_args = get_args()
        rlhf_args = global_args.rlhf_args
        resource_manager = ResourceManager(models)
        model_manager = ModelManager(models, resource_manager, global_args)
        self.remote_models = model_manager.remote()
        self.rlhf_args = rlhf_args


    def setup(self):
        for model in self.remote_models:
            ray.get(model.setup())
        

    @property
    def models(self):
        return self.remote_models



class RLHFEngine(Engine):
    """rlhf engine"""

    def __init__(self,
                 policy: RLHFModelWrapper,
                 reference: RLHFModelWrapper,
                 reward: RLHFModelWrapper,
                 value: RLHFModelWrapper,
                 ppo_policy: RLHFModelWrapper,
                 ppo_value: RLHFModelWrapper):
        super().__init__(policy, reference, reward, value, ppo_policy, ppo_value)
        policy, reference, reward, value, ppo_policy, ppo_value = self.remote_models
        self.env = PPOEnv(self.rlhf_args, policy, reference, reward, value)
        self.trainer = PPOTrainer(self.rlhf_args, ppo_policy, ppo_value)

    def set_trainer(self, trainer):
        self.trainer = trainer
        return self

    def set_environment(self, env):
        self.env = env
        return self

    def learn(self):
        self.env.setup()
        self.trainer.setup()
        for iter in range(self.rlhf_args.num_ppo_iteration):
            ppo_data_loader = self.env.make_experiences()
            self.trainer.train(ppo_data_loader)
