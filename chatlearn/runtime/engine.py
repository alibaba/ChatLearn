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

import math
import torch

from chatlearn.checkpoint.checkpoint_manager import CheckpointManager
from chatlearn.data.data import StreamDataset
from chatlearn.models.rlhf_module import RLHFModule
from chatlearn.runtime.environment import PPOEnv
from chatlearn.runtime.evaluator import Evaluator
from chatlearn.runtime.trainer import PPOTrainer
from chatlearn.schedule.model_manager import ModelManager
from chatlearn.schedule.resource_manager import ResourceManager
from chatlearn.utils import future
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.utils.timer import Timers

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
        """
        :meta private:
        """
        # include compile in init, compile dependencies need to be called serially
        for model in self.remote_models:
            model.init()
        # do not include compile dependencies in setup
        refs = []
        refs_val = []
        for model in self.remote_models:
            refs += model.model_setup()
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
            logger.info(f"{LOG_START} [{model.name}] {summary[-1]}")
        self.logging_memory()

    def stop(self):
        self.model_manager.clean()


class RLHFEngine(Engine):
    """
    RLHF engine.

    Args
    ----
    policy : RLHFModule
        policy inference model
    reference : RLHFModule
        reference inference model
    reward : RLHFModule
        reward inference model
    value : RLHFModule
        value inference model
    ppo_policy : RLHFModule
        ppo policy training model
    ppo_value : RLHFModule
        ppo value training model
    """

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
        self._wrap_data = True
        self._relay_sample_fn = None
        self._ppo_data_loader = None

    def _create_remote_models(self):
        """
        :meta private:
        """
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
        """
        :meta private:
        """
        super().setup()
        self.env = self.create_env(self.policy, self.reference, self.reward, self.value)
        self.env.set_dataset(self._dataset, self._drop_last, self._wrap_data)
        self.trainer = self.create_trainer(self.ppo_policy, self.ppo_value)
        if self.evaluator is not None:
            self.evaluator.update_models(self.remote_models)
        self.model_manager.build_parameter_group()
        self.model_manager.start_error_monitor()

    def create_env(self, policy, reference, reward, value):
        """
        :meta private:
        """
        env = PPOEnv(self.rlhf_args,
                     policy,
                     reference,
                     reward,
                     value)
        return env

    def create_trainer(self, ppo_policy, ppo_value):
        """
        :meta private:
        """
        return PPOTrainer(self.rlhf_args, ppo_policy.replicas[0], ppo_value.replicas[0])

    def set_dataset(self, dataset, drop_last=False):
        """
        Set prompt dataset.

        Args
        ----
        dataset : list
            a list of prompt string
        drop_last : bool
            drop last samples if dataset is indivisible by `sample_per_episode`
        """
        self._dataset = dataset
        if not self.rlhf_args.enable_indivisible_batch_size:
            # When enable_indivisible_batch_size is set to False, the dataloader either discards
            # the last incomplete batch or complements the last batch.
            self._drop_last = drop_last
            self._wrap_data = not drop_last
        else:
            if drop_last:
                logger.warning("when enable_indivisible_batch_size is set to True, drop_last will be specified as False")
            self._drop_last = False
            self._wrap_data = False

    def set_evaluator(self, evaluator):
        """
        Set model evaluator.

        Args
        ----
        evaluator - Evaluator
        """
        self.evaluator = evaluator

    def logging_summary(self, iteration=-1):
        """
        :meta private:
        """
        super().logging_summary(iteration)
        episode_str, episode_stats = self.timers.log(names=['episode', 'sync_parameters'], return_dict=True)
        logger.info(f"{LOG_START} RLHF episode summary episode iteration {iteration} {episode_str}")
        self.episode_stats = episode_stats
        return episode_stats

    def set_relay_sample_fn(self, relay_sample_fn):
        """
        Set custom relay_sample_fn.

        Args
        ----
            relay_sample_fn: inputs List[EpisodeRelayBuffer], return a list of dict.
        """
        self._relay_sample_fn = relay_sample_fn

    def learn(self):
        """
        Start rlhf training.
        """
        self.timers("rlhf").start()
        self.timers("setup").start()
        self.setup()
        self.trainer.setup(self.model_manager.model_packs)
        self.env.setup(self.model_manager.model_packs)
        if self.evaluator:
            self.evaluator.setup(self.model_manager.model_packs)
        self.timers("setup").stop()
        logger.info(f"{LOG_START} RLHF setup summary {self.timers.log(names=['setup'])}")
        self.logging_memory()
        self._resume_from_data_checkpoint()

        ppo_data_loader = StreamDataset.remote(self.rlhf_args.stream_data_loader_type,
                                               self.rlhf_args.train_micro_batch_size,
                                               self.env._padding_config,
                                               self.rlhf_args.max_relay_episode)
        self.model_manager.sync_parameters(requires_grad=False)
        self._ppo_data_loader = ppo_data_loader
        for episode_id in range(self._start_episode, self.rlhf_args.num_ppo_episode):
            if self.rlhf_args.nsys:
                if episode_id == 4:
                    torch.cuda.cudart().cudaProfilerStart()
                if episode_id == 5:
                    torch.cuda.cudart().cudaProfilerStop()
            self.timers("episode").start()
            self.before_episode()
            logger.info(f"start train episode_id: {episode_id + 1}/{self.rlhf_args.num_ppo_episode}")
            queue = self.env.make_experiences()
            refs = ppo_data_loader.set_dataset.remote(queue, episode_id, self._relay_sample_fn, self.rlhf_args.sample_per_episode)
            future.wait(refs)
            self.trainer.set_data_loader(ppo_data_loader)
            logger.info("set dataloader for trainer done")
            self.trainer.train(episode_id)
            logger.info(f"train episode_id: {episode_id + 1}/{self.rlhf_args.num_ppo_episode} done")
            self.timers("sync_parameters").start()
            self.model_manager.sync_parameters()
            self.timers("sync_parameters").stop()
            logger.info(f"train episode_id: {episode_id + 1}/{self.rlhf_args.num_ppo_episode} parameter sync done")
            self.after_episode()
            self.timers("episode").stop()
            self.logging_summary(episode_id)
            self.save_checkpoint(episode_id)
            self.evaluate(episode_id)

        self.timers("rlhf").stop()
        logger.info(f"{LOG_START} RLHF overall summary {self.timers.log(names=['rlhf'])}")
        logger.info("train rlhf done")

    def _resume_from_data_checkpoint(self):
        if self.rlhf_args.data_checkpoint_path:
            data_ckpt_manager = CheckpointManager(self.policy.replicas[0], self.rlhf_args.data_checkpoint_path,
                                                  self.rlhf_args.max_data_ckpt_nums,
                                                  self.rlhf_args.load_data_checkpoint_iteration)
            meta = data_ckpt_manager.resume_meta()
            if meta:
                self._start_episode = meta["episode"] + 1
                self.trainer.iteration = meta["train_iteration"]
                for model in self.remote_models:
                    start_iteration = self._start_episode * \
                        math.ceil(self.rlhf_args.sample_per_episode / model.num_replica / model.module_args.generation_batch_size)
                    logger.info(f"Set start iteration {start_iteration} for model {model.name}")
                    model.set_start_iteration(start_iteration)

    def save_checkpoint(self, episode_id):
        """
        :meta private:
        """
        if self.rlhf_args.save_episode_interval and \
            (episode_id + 1) % self.rlhf_args.save_episode_interval == 0:
            ref0 = self.ppo_policy.replicas[0].save_checkpoint(self.trainer.iteration)
            ref1 = self.ppo_value.replicas[0].save_checkpoint(self.trainer.iteration)
            refs = [ref0, ref1]
            for i, model in enumerate(self.policy.replicas):
                refs.append(model.save_data_checkpoint(i, self.trainer.iteration, episode_id))
            future.get(refs)
            logger.info(f"save checkpoint episode {episode_id}, train iteration {self.trainer.iteration} done")

    def evaluate(self, episode_id):
        """
        :meta private:
        """
        if self.evaluator is not None and \
            self.rlhf_args.eval_episode_interval and \
            (episode_id + 1) % self.rlhf_args.eval_episode_interval == 0:
            logger.info("start evaluate")
            self.timers("evaluate").start()
            self.evaluator.eval(episode_id, self.trainer.iteration)
            self.timers("evaluate").stop()
            super().logging_summary(episode_id)
            logger.info(f"evaluate done {self.timers.log(names=['evaluate'])}")


class EvalEngine(Engine):
    """Evaluate Engine"""

    def __init__(self, models):
        if not isinstance(models, list):
            models = [models]
        super().__init__(*models)
        self.evaluator = Evaluator(self.remote_models, self.rlhf_args)

    def set_dataset(self, dataset):
        """
        Set prompt dataset.

        Args
        ----
        dataset : list
            a list of prompt string
        """
        self.evaluator.set_dataset(dataset)

    def eval(self):
        """
        Start evaluating.
        """
        self.setup()
        self.evaluator.setup(self.model_manager.model_packs)
        queue = self.evaluator.eval()
        return queue
