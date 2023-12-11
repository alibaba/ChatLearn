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

from chatlearn.checkpoint.checkpoint_manager import CheckpointManager
from chatlearn.data.data import StreamDataset
from chatlearn.models.rlhf_module import RLHFModule
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.evaluator import Evaluator
from chatlearn.runtime.trainer import Trainer
from chatlearn.schedule.model_manager import ModelManager
from chatlearn.schedule.resource_manager import ResourceManager
from chatlearn.utils import future
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.utils.timer import Timers

LOG_START = ">>>>>>>>>>>"


class BaseEngine:
    """Base Engine"""

    def __init__(self, *models):
        self._models = models
        self.global_args = get_args()
        self.rlhf_args = self.global_args.rlhf_args
        self.timers = Timers()

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
        self._create_remote_models()
        # for ease to access model by self.{model_name}
        for model in self.remote_models:
            setattr(self, model.name, model)
        # include compile in init, compile dependencies need to be called serially
        for model in self.remote_models:
            model.init()
        # do not include compile dependencies in setup
        # if the program hang in setup, may try to set concurrent_setup to False.
        if self.rlhf_args.concurrent_setup:
            refs = []
            refs_val = []
            for model in self.remote_models:
                refs += model.model_setup()
                refs_val += model.validate()
            future.wait(refs)
            future.wait(refs_val)
        else:
            for model in self.remote_models:
                future.wait(model.model_setup())
                future.wait(model.validate())
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


class Engine(BaseEngine):
    """Engine"""

    def __init__(self, environment=None, trainer=None, evaluator=None):
        """
        Engine.

        Args
        ----
        environment : Environment
        trainer : Trainer
        evaluator: Evaluator
        """
        models = []
        for executor in [environment, trainer, evaluator]:
            if executor:
                for model in executor.models:
                    if model not in models:
                        models.append(model)
        super().__init__(*models)
        self.env = environment
        self.trainer = trainer
        self.evaluator = evaluator
        self._start_episode = 0
        self._dataset = None
        self._drop_last = False
        self._wrap_data = True
        self._relay_sample_fn = None
        self._ppo_data_loader = None
        self._param_sync_pairs = []

    def set_parameter_sync(self, src_model, dst_model):
        """
        sync model parameter from src_model to dst_model

        Args
        ----
        src_model: RLHFModule
            src model to sync parameters
        dst_model: RLHFModule
            destination model to sync parameters
        """
        self._param_sync_pairs.append((src_model, dst_model))
        dst_model.set_src_parameter_model(src_model)
        return self

    def _create_remote_models(self):
        """
        :meta private:
        """
        resource_manager = ResourceManager(self._models)
        self.model_manager = ModelManager(self._models, resource_manager, self.global_args)
        for src_model, dst_model in self._param_sync_pairs:
            self.model_manager.set_parameter_sync(src_model, dst_model)
        self.model_manager.remote()
        self.remote_models = self.model_manager.dist_models
        self.named_models = {model.name: model for model in self.remote_models}

    def setup(self):
        """
        :meta private:
        """
        super().setup()
        self._executors = [self.env, self.trainer, self.evaluator]
        for executor in self._executors:
            if executor:
                executor.update_models(self.remote_models)
        if self.env:
            self.env.set_dataset(self._dataset)
        self.model_manager.build_parameter_group()
        self.model_manager.start_error_monitor()

    def set_dataset(self, dataset):
        """
        Set prompt dataset.

        Args
        ----
        dataset : list
            a list of prompt string
        """
        self._dataset = dataset
        return self

    def set_trainer(self, trainer):
        self.trainer = trainer
        return self

    def set_environment(self, env):
        self.env = env
        return self

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator
        return self

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
        self.timers("rlhf").start()
        self.timers("setup").start()
        self.setup()
        for executor in self._executors:
            if executor:
                executor.setup()
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
            refs = ppo_data_loader.set_dataset.remote(queue, episode_id, self._relay_sample_fn,
                                                      self.rlhf_args.sample_per_episode)
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
            data_ckpt_manager = CheckpointManager(self.models[0].replicas[0], self.rlhf_args.data_checkpoint_path,
                                                  self.rlhf_args.max_data_ckpt_nums,
                                                  self.rlhf_args.load_data_checkpoint_iteration)
            if self.rlhf_args.enable_resume_training:
                meta = data_ckpt_manager.resume_meta()
                if meta:
                    self._start_episode = meta["episode"] + 1
                    self.trainer.iteration = meta["train_iteration"]
                    if self.trainer.iteration > 0:
                        logger.info(f"ChatLearn continue train with meta {meta}")

    def save_checkpoint(self, episode_id):
        """
        :meta private:
        """
        if self.rlhf_args.save_episode_interval and \
                (episode_id + 1) % self.rlhf_args.save_episode_interval == 0:
            refs = []
            for model in self.trainer.models:
                refs.append(model.replicas[0].save_checkpoint(self.trainer.iteration))
            for i, model in enumerate(self.models[0].replicas):
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


class RLHFEngine(Engine):
    """RLHFEngine"""

    def __init__(self,
                 policy: RLHFModule,
                 reference: RLHFModule,
                 reward: RLHFModule,
                 value: RLHFModule,
                 ppo_policy: RLHFModule,
                 ppo_value: RLHFModule):
        def env_compute_flow(batch):
            policy_out = policy.forward_step(batch)
            ref_out = reference.forward_step(policy_out)
            value_out = value.forward_step(policy_out)
            reward_out = reward.forward_step(policy_out, ref_out, value_out)
            return value_out, reward_out

        def trainer_compute_flow(batch):
            ppo_policy.train_step(batch)
            ppo_value.train_step(batch)

        env = Environment([policy, reference, value, reward]).set_flow(env_compute_flow)
        trainer = Trainer([ppo_policy, ppo_value]).set_flow(trainer_compute_flow)
        super().__init__(env, trainer)
        self.set_parameter_sync(ppo_policy, policy)
        self.set_parameter_sync(ppo_value, value)


class EvalEngine(Engine):
    """Evaluation Engine"""

    def __init__(self, models):
        evaluator = Evaluator(models)
        super().__init__(evaluator=evaluator)

    def setup(self):
        super().setup()
        self.evaluator.set_dataset(self._dataset)

    def set_dataset(self, dataset):
        """
        Set prompt dataset.

        Args
        ----
        dataset : list
            a list of prompt string
        """
        self._dataset = dataset

    def eval(self):
        """
        Start evaluating.
        """
        self.setup()
        self.evaluator.setup()
        queue = self.evaluator.eval()
        return queue
