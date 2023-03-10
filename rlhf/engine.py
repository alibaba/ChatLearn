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
        self.model_manager = ModelManager(models, resource_manager, global_args)
        self.remote_models = self.model_manager.remote_models
        self.named_models = {model.name: model for model in self.remote_models}
        self.rlhf_args = rlhf_args


    def setup(self):
        for model in self.remote_models:
            status = ray.get(model.setup())
            print(f"setup model {model.name} done, status: {status}", flush=True)
        print("done setup all models", flush=True)
        

    @property
    def models(self):
        return self.remote_models

    def get_model(self, name):
        return self.named_models[name]



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
        self.policy, self.reference, self.reward, self.value, self.ppo_policy, self.ppo_value = \
                policy, reference, reward, value, ppo_policy, ppo_value


    def setup(self):
        super().setup()
        self.model_manager.set_model_sync(self.ppo_policy, self.policy)
        self.model_manager.set_model_sync(self.ppo_value, self.value)
        self.model_manager.start_error_monitor()


    def set_dataset(self, dataset):
        self.env.set_dataset(dataset)

    def set_trainer(self, trainer):
        self.trainer = trainer
        return self


    def set_environment(self, env):
        self.env = env
        return self


    def learn(self):
        self.setup()
        self.env.setup()

        for ppo_iter in range(self.rlhf_args.num_ppo_iteration):
            print(f"start train ppo_iter: {ppo_iter+1}/{self.rlhf_args.num_ppo_iteration}", flush=True)
            ppo_data_loader = self.env.make_experiences()
            self.trainer.set_data_loader(ppo_data_loader)
            self.trainer.train()
            print(f"train ppo_iter: {ppo_iter+1}/{self.rlhf_args.num_ppo_iteration} done", flush=True)
            self.model_manager.sync_parameters()
            print(f"train ppo_iter: {ppo_iter+1}/{self.rlhf_args.num_ppo_iteration} parameter sync done", flush=True)
        self.model_manager.clean()
