import ray
from ray.util.queue import Queue
from rlhf.model_manager import ModelManager
from rlhf.resource import ResourceManager
from rlhf.model_wrapper import RLHFModelWrapper
from rlhf.environment import PPOEnv
from rlhf.trainer import PPOTrainer
from rlhf.global_vars import get_args
from rlhf.logger import logger
from rlhf.data import StreamDataset, RLHFDataLoader
from rlhf import utils


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
            status = utils.get(model.setup())
            logger.info(f"setup model {model.name} done, status: {status}")
            status = utils.get(model.validate())
            logger.info(f"validate model {model.name} done, status: {status}")
        logger.info("done setup all models")
        

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
        self.envs = self.create_env(policy, reference, reward, value)
        self.trainer = self.create_trainer(ppo_policy, ppo_value)
        self.policy, self.reference, self.reward, self.value, self.ppo_policy, self.ppo_value = \
                policy, reference, reward, value, ppo_policy, ppo_value


    def setup(self):
        super().setup()
        self.model_manager.set_model_sync(self.ppo_policy, self.policy)
        self.model_manager.set_model_sync(self.ppo_value, self.value)
        self.model_manager.start_error_monitor()


    def create_env(self, policy, reference, reward, value):
        envs = []
        for i in range(self.rlhf_args.num_rollout_worker):
            env = PPOEnv(self.rlhf_args,
                         policy.replicas[i],
                         reference.replicas[i],
                         reward.replicas[i],
                         value.replicas[i])
            envs.append(env)
        return envs

    def create_trainer(self, ppo_policy, ppo_value):
        return PPOTrainer(self.rlhf_args, ppo_policy.replicas[0], ppo_value.replicas[0])


    def set_dataset(self, dataset):
        # TODO: compare with use only master dataloader
        data_len = len(dataset)
        indices = utils.split_index(data_len, self.rlhf_args.num_rollout_worker)

        for i, (start, end) in enumerate(indices):
            data_part = dataset[start:end]
            self.envs[i].set_dataset(data_part)


    def set_trainer(self, trainer):
        self.trainer = trainer
        return self


    def learn(self):
        self.setup()
        self.trainer.setup()
        for env in self.envs:
            env.setup()

        for ppo_iter in range(self.rlhf_args.num_ppo_iteration):
            queue = Queue()
            logger.info(f"start train ppo_iter: {ppo_iter+1}/{self.rlhf_args.num_ppo_iteration}")
            for i in range(self.rlhf_args.num_rollout_worker):
                self.envs[i].make_experiences(queue)
            ppo_data_loader = StreamDataset.remote(queue, self.rlhf_args.sample_per_episode,
                                                   self.rlhf_args.train_global_batch_size,
                                                   self.envs[0]._padding_config, cache=True)
            self.trainer.set_data_loader(ppo_data_loader)
            self.trainer.train()
            logger.info(f"train ppo_iter: {ppo_iter+1}/{self.rlhf_args.num_ppo_iteration} done")
            # TODO: overlap
            self.model_manager.sync_parameters()
            logger.info(f"train ppo_iter: {ppo_iter+1}/{self.rlhf_args.num_ppo_iteration} parameter sync done")
        self.model_manager.clean()
