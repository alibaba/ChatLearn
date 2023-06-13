# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Engine"""

import torch

from rlhf.checkpoint.checkpoint_manager import CheckpointManager
from rlhf.data.data import StreamDataset
from rlhf.models.rlhf_module import RLHFModule
from rlhf.runtime.environment import PPOEnv
from rlhf.runtime.evaluator import Evaluator
from rlhf.runtime.trainer import PPOTrainer
from rlhf.schedule.model_manager import ModelManager
from rlhf.schedule.resource import ResourceManager
from rlhf.utils import future
from rlhf.utils.global_vars import get_args
from rlhf.utils.logger import logger
from rlhf.utils.timer import Timers

LOG_START = ">>>>>>>>>>>"


class Engine:
    """Base Engine"""

    def __init__(self, *models):
        self._models = models
        self.global_args = get_args()
        self.rlhf_args = self.global_args.rlhf_args
        self.timers = Timers()
        self._create_remote_models()

    def _create_remote_models(self):
        resource_manager = ResourceManager(self._models)
        self.model_manager = ModelManager(self._models, resource_manager, self.global_args)
        self.model_manager.remote()
        self.remote_models = self.model_manager.dist_models
        self.named_models = {model.name: model for model in self.remote_models}

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
        future.wait(refs)
        future.wait(refs_val)
        logger.info("done setup all models")

    def before_episode(self):
        for model in self.remote_models:
            future.get(model.before_episode())

    def after_episode(self):
        for model in self.remote_models:
            future.get(model.after_episode())

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
        summaries = future.get(refs)

        logger.info(f"{LOG_START} memory summary:")
        for model, summary in zip(self.remote_models, summaries):
            mem_str = ' | '.join(['{:.2f}'.format(i) for i in flatten(summary)])
            mem_log = f"peak_mem(GB): {mem_str}"
            logger.info(f"{LOG_START} {model.name} {mem_log}")

    def logging_summary(self, iteration=-1):
        refs = []
        for model in self.remote_models:
            time_ref = model.replicas[0].timer_summary()
            refs.append(time_ref)
        summaries = future.get(refs)

        logger.info(f"{LOG_START} PPO iteration {iteration} time summary for each model as follows:")
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
        reward.module_args.return_rlhf_data = True
        super().__init__(policy, reference, reward, value, ppo_policy, ppo_value)
        self.evaluator = None
        self._start_episode = 0
        self._dataset = None
        self._drop_last = False

    def _create_remote_models(self):
        resource_manager = ResourceManager(self._models)
        self.model_manager = ModelManager(self._models, resource_manager, self.global_args)
        ppo_policy = self._models[4]
        policy = self._models[0]
        ppo_value = self._models[5]
        value = self._models[3]
        self.model_manager.set_model_sync(ppo_policy, policy)
        self.model_manager.set_model_sync(ppo_value, value)
        self.model_manager.remote()
        self.remote_models = self.model_manager.dist_models
        self.named_models = {model.name: model for model in self.remote_models}
        self.policy, self.reference, self.reward, self.value, self.ppo_policy, self.ppo_value = self.remote_models

    def setup(self):
        super().setup()
        self.env = self.create_env(self.policy, self.reference, self.reward, self.value)
        self.env.set_dataset(self._dataset, self._drop_last)
        self.trainer = self.create_trainer(self.ppo_policy, self.ppo_value)
        if self.evaluator is not None:
            self.evaluator.update_models(self.remote_models)
        self.model_manager.build_parameter_group()
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
        self._dataset = dataset
        self._drop_last = drop_last

    def set_trainer(self, trainer):
        self.trainer = trainer
        return self

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def logging_summary(self, iteration=-1):
        super().logging_summary(iteration)
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
        if self.evaluator:
            self.evaluator.setup()
        self.timers("setup").stop()
        logger.info(f"{LOG_START} RLHF setup summary {self.timers.log(names=['setup'])}")
        self.logging_memory()
        self.resume_from_data_checkpoint()
        for ppo_iter in range(self._start_episode, self.rlhf_args.num_ppo_episode):
            if self.rlhf_args.nsys:
                if ppo_iter == 4:
                    torch.cuda.cudart().cudaProfilerStart()
                if ppo_iter == 5:
                    torch.cuda.cudart().cudaProfilerStop()
            self.timers("episode").start()
            self.before_episode()
            logger.info(f"start train ppo_iter: {ppo_iter + 1}/{self.rlhf_args.num_ppo_episode}")
            queue = self.env.make_experiences()
            if self.rlhf_args.dynamic_train_samples:
                sample_per_episode = 0
            else:
                sample_per_episode = self.rlhf_args.sample_per_episode
            ppo_data_loader = StreamDataset.remote(queue, sample_per_episode,
                                                   self.rlhf_args.train_micro_batch_size,
                                                   self.env._padding_config, cache=True)
            self.trainer.set_data_loader(ppo_data_loader)
            logger.info("set dataloader for trainer done")
            self.trainer.train(ppo_iter)
            logger.info(f"train ppo_iter: {ppo_iter + 1}/{self.rlhf_args.num_ppo_episode} done")
            self.timers("sync_parameters").start()
            self.model_manager.sync_parameters()
            self.timers("sync_parameters").stop()
            logger.info(f"train ppo_iter: {ppo_iter + 1}/{self.rlhf_args.num_ppo_episode} parameter sync done")
            self.after_episode()
            self.timers("episode").stop()
            self.logging_summary(ppo_iter)
            self.save_checkpoint(ppo_iter)
            self.evaluate(ppo_iter)

        self.timers("rlhf").stop()
        logger.info(f"{LOG_START} RLHF overall summary {self.timers.log(names=['rlhf'])}")
        logger.info("train rlhf done")
        self.model_manager.clean()

    def resume_from_data_checkpoint(self):
        if self.rlhf_args.data_checkpoint_path:
            data_ckpt_manager = CheckpointManager(self.policy.replicas[0], self.rlhf_args.data_checkpoint_path,
                                                  self.rlhf_args.max_data_ckpt_nums,
                                                  self.rlhf_args.load_data_checkpoint_iteration)
            meta = data_ckpt_manager.resume_meta()
            if meta:
                self._start_episode = meta["episode"] + 1
                self.trainer.iteration = meta["train_iteration"]

    def save_checkpoint(self, ppo_iter):
        if self.rlhf_args.save_episode_interval and \
            (ppo_iter + 1) % self.rlhf_args.save_episode_interval == 0:
            ref0 = self.ppo_policy.replicas[0].save_checkpoint(self.trainer.iteration)
            ref1 = self.ppo_value.replicas[0].save_checkpoint(self.trainer.iteration)
            refs = [ref0, ref1]
            for i, model in enumerate(self.policy.replicas):
                refs.append(model.save_data_checkpoint(i, self.trainer.iteration, ppo_iter))
            future.get(refs)
            logger.info(f"save checkpoint episode {ppo_iter}, train iteration {self.trainer.iteration} done")

    def evaluate(self, ppo_iter):

        if self.evaluator is not None and \
            self.rlhf_args.eval_episode_interval and \
            (ppo_iter + 1) % self.rlhf_args.eval_episode_interval == 0:
            logger.info("start evaluate")
            self.timers("evaluate").start()
            self.evaluator.eval(ppo_iter, self.trainer.iteration)
            self.timers("evaluate").stop()
            super().logging_summary(ppo_iter)
            logger.info(f"evaluate done {self.timers.log(names=['evaluate'])}")


class EvalEngine(Engine):
    """Evaluate Engine"""

    def __init__(self, models):
        if not isinstance(models, list):
            models = [models]
        super().__init__(*models)
        self.evaluator = Evaluator(self.remote_models, self.rlhf_args)

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
        queue = self.evaluator.eval()
        return queue
    
    def stop(self):
        self.model_manager.clean()
