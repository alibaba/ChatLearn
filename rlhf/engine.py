import ray
from ray.util.queue import Queue
from rlhf.model_manager import ModelManager
from rlhf.resource import ResourceManager
from rlhf.model_wrapper import RLHFModule
from rlhf.environment import PPOEnv
from rlhf.trainer import PPOTrainer
from rlhf.global_vars import get_args
from rlhf.logger import logger
from rlhf.data import StreamDataset, RLHFDataLoader
from rlhf import utils
from rlhf.timer import Timers
from rlhf.evaluator import Evaluator


LOG_START = ">>>>>>>>>>>"

class Engine:

    def __init__(self, *models):
        global_args = get_args()
        rlhf_args = global_args.rlhf_args
        resource_manager = ResourceManager(models, rlhf_args.colocation)
        self.model_manager = ModelManager(models, resource_manager, global_args)
        self.remote_models = self.model_manager.remote_models
        self.named_models = {model.name: model for model in self.remote_models}
        self.rlhf_args = rlhf_args
        self.timers = Timers()


    def setup(self):
        # include compile in init, compile dependencies need to be called serially
        for model in self.remote_models:
            model.init()
        # do not include compile dependencies in setup
        refs = []
        refs_val = []
        for model in self.remote_models:
            refs += model.setup()
            refs_val += model.validate()
        utils.get(refs)
        utils.get(refs_val)
        logger.info("done setup all models")


    def before_episode(self):
        for model in self.remote_models:
            utils.get(model.before_episode()) 
    

    def after_episode(self):
        for model in self.remote_models:
            utils.get(model.after_episode())


    @property
    def models(self):
        return self.remote_models

    def get_model(self, name):
        return self.named_models[name]


    def logging_memory(self):
        def flatten(xs):
            for x in xs:
                if isinstance(x, list):
                    yield from flatten(x)
                else:
                    yield x

        refs = []
        for model in self.remote_models:
            mem_ref = model.peak_memory()
            refs.append(mem_ref)
        summaries = utils.get(refs)

        logger.info(f"{LOG_START} memory summary:")
        for model, summary in zip(self.remote_models, summaries):
            mem_str = ' | '.join(['{:.2f}'.format(i) for i in flatten(summary)])
            mem_log = f"peak_mem(GB): {mem_str}"
            logger.info(f"{LOG_START} {model.name} {mem_log}")


    def logging_summary(self):
        refs = []
        for model in self.remote_models:
            time_ref = model.replicas[0].timer_summary()
            refs.append(time_ref)
        summaries = utils.get(refs)

        logger.info(f"{LOG_START} time summary for each model as follows:")
        for model, summary in zip(self.remote_models, summaries):
            logger.info(f"{LOG_START} [{model.name}] {summary[0]}")
        self.logging_memory()



class RLHFEngine(Engine):
    """rlhf engine"""

    def __init__(self,
                 policy: RLHFModule,
                 reference: RLHFModule,
                 reward: RLHFModule,
                 value: RLHFModule,
                 ppo_policy: RLHFModule,
                 ppo_value: RLHFModule):
        for model in [policy, reference, reward, value]:
            model.register_func("forward_step")
        for model in [ppo_policy, ppo_value]:
            model.register_func("train_step")
        super().__init__(policy, reference, reward, value, ppo_policy, ppo_value)
        policy, reference, reward, value, ppo_policy, ppo_value = self.remote_models
        self.env = self.create_env(policy, reference, reward, value)
        self.trainer = self.create_trainer(ppo_policy, ppo_value)
        self.policy, self.reference, self.reward, self.value, self.ppo_policy, self.ppo_value = \
                policy, reference, reward, value, ppo_policy, ppo_value



    def setup(self):
        super().setup()
        self.model_manager.set_model_sync(self.ppo_policy, self.policy)
        self.model_manager.set_model_sync(self.ppo_value, self.value)
        self.model_manager.start_error_monitor()


    def create_env(self, policy, reference, reward, value):
        env = PPOEnv(self.rlhf_args,
                     policy,
                     reference,
                     reward,
                     value)
        return env

    def create_trainer(self, ppo_policy, ppo_value):
        return PPOTrainer(self.rlhf_args, ppo_policy.replicas[0], ppo_value.replicas[0])


    def set_dataset(self, dataset, drop_last=False):
        self.env.set_dataset(dataset, drop_last)


    def set_trainer(self, trainer):
        self.trainer = trainer
        return self


    def logging_summary(self, iteration):
        super().logging_summary()
        episode_str, episode_stats = self.timers.log(names=['episode', 'sync_parameters'], return_dict=True)
        logger.info(f"{LOG_START} RLHF episode summary episode iteration {iteration} {episode_str}")
        self.episode_stats = episode_stats
        return episode_stats


    def learn(self):
        self.timers("rlhf").start()
        self.timers("setup").start()
        self.setup()
        self.trainer.setup()
        self.env.setup()
        self.timers("setup").stop()
        logger.info(f"{LOG_START} RLHF setup summary {self.timers.log(names=['setup'])}")
        self.logging_memory()

        for ppo_iter in range(self.rlhf_args.num_ppo_episode):
            self.timers("episode").start()
            self.before_episode()
            logger.info(f"start train ppo_iter: {ppo_iter+1}/{self.rlhf_args.num_ppo_episode}")
            queue = self.env.make_experiences()
            ppo_data_loader = StreamDataset.remote(queue, self.rlhf_args.sample_per_episode,
                                                   self.rlhf_args.train_micro_batch_size,
                                                   self.env._padding_config, cache=True)
            self.trainer.set_data_loader(ppo_data_loader)
            logger.info(f"set dataloader for environment done")
            self.trainer.train(ppo_iter)
            logger.info(f"train ppo_iter: {ppo_iter+1}/{self.rlhf_args.num_ppo_episode} done")
            self.timers("sync_parameters").start()
            self.model_manager.sync_parameters()
            self.timers("sync_parameters").stop()
            logger.info(f"train ppo_iter: {ppo_iter+1}/{self.rlhf_args.num_ppo_episode} parameter sync done")
            self.after_episode()
            self.timers("episode").stop()
            self.logging_summary(ppo_iter)
        self.timers("rlhf").stop()
        logger.info(f"{LOG_START} RLHF overall summary {self.timers.log(names=['rlhf'])}")
        logger.info("train rlhf done")
        self.model_manager.clean()


class EvalEngine(Engine):

    def __init__(self, models):
        if not isinstance(models, list):
            models = [models]
        super().__init__(*models)
        self.evaluator = Evaluator(self.rlhf_args, self.remote_models)


    def set_dataset(self, dataset):
        self.evaluator.set_dataset(dataset)



    def register_func(self, model_name, func_name):
        """
        register eval func for certain model, the default eval func is eval_step
        """
        self.evaluator.register_func(model_name, func_name)


    def eval(self):
        self.setup()
        self.evaluator.setup()
        queue = Queue()
        self.evaluator.eval(queue)
        # end of evaluation
        queue.put(None)
        return queue
