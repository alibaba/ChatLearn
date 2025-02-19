# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
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

import os
import shutil
import torch

from chatlearn.checkpoint.checkpoint_manager import CheckpointManager
from chatlearn.data.data import StreamDataset
from chatlearn.models.base_module import BaseModule
from chatlearn.runtime.dist_actor import DistVLLMActor
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.evaluator import Evaluator
from chatlearn.runtime.trainer import Trainer
from chatlearn.schedule.model_manager import ModelManager
from chatlearn.schedule.resource_manager import ResourceManager
from chatlearn.utils import future
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.utils.utils import get_full_proc_memory_info
from chatlearn.utils.timer import Timers

LOG_START = ">>>>>>>>>>>"


class BaseEngine:
    """Base Engine"""

    def __init__(self, *models):
        self._models = models
        self.global_args = get_args()
        self.runtime_args = self.global_args.runtime_args
        self._timers = Timers()

    def set_timers(self, _timers):
        self._timers = _timers

    @property
    def timers(self):
        return self._timers

    def timer_summary(self):
        """
        :meta private:
        """
        if self._timers:
            return self._timers.log(reset=False, return_dict=True)

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

        if hasattr(self, '_param_sync_pairs'):
            ref_set_src = []
            for src_model, dst_model in self._param_sync_pairs:
                remote_src_model = getattr(self, src_model.name)
                remote_dst_model = getattr(self, dst_model.name)
                ref_set_src += remote_dst_model.set_src_parameter_model(remote_src_model)
            future.wait(ref_set_src)
        # include compile in init, compile dependencies need to be called serially
        logger.info(get_full_proc_memory_info('Before model init'))
        for model in self.remote_models:
            model.init()
        logger.info(get_full_proc_memory_info('After model init'))
        # do not include compile dependencies in setup
        # if the program hang in setup, may try to set concurrent_setup to False.
        self.timers("setup_models").start()
        if self.runtime_args.concurrent_setup:
            refs = []
            refs_val = []
            for model in self.remote_models:
                refs += model.model_setup()
                refs_val += model.validate()
            future.wait(refs)
            future.wait(refs_val)
        else:
            for model in self.remote_models:
                logger.info(f"start setup and validate {model.name}")
                future.wait(model.model_setup())
                future.wait(model.validate())
                logger.info(f"done setup and validate {model.name}")
        self.timers("setup_models").stop()
        logger.info(
            f"{LOG_START} setup_models summary {self.timers.log(names=['setup_models'])}")

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

        logger.debug(f"{LOG_START} memory summary:")
        for model, summary in zip(self.remote_models, summaries):
            mem_str = ' | '.join(['{:.2f}'.format(i) for i in flatten(summary)])
            mem_log = f"peak_mem(GiB): {mem_str}"
            logger.debug(f"{LOG_START} {model.name} {mem_log}")

    def logging_summary(self, iteration=-1):
        _, e2e_time_dict = self.timer_summary()
        refs = []
        for model in self.remote_models:
            time_ref = model.replicas[0].timer_summary(e2e_cost=e2e_time_dict.get(model.name, None))
            refs.append(time_ref)
        summaries = future.get(refs)

        logger.info(f"{LOG_START} episode iteration {iteration + 1} time summary for each model as follows:")
        for model, summary in zip(self.remote_models, summaries):
            summary = summary[-1] if isinstance(summary, list) else summary
            logger.info(f"{LOG_START} [{model.name}] {summary}")
        self.logging_memory()

    def stop(self):
        self.model_manager.clean()


class Engine(BaseEngine):
    """Engine"""

    def __init__(self, environment=None, trainer=None, evaluator=None, name='alignment'):
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
        if environment:
            environment.set_timers(self.timers)
        if trainer:
            trainer.set_timers(self.timers)
        self.env = environment
        self.trainer = trainer
        self.evaluator = evaluator
        self._start_episode = 0
        self._dataset = None
        self._post_process_func = None
        self._drop_last = False
        self._wrap_data = True
        self._relay_sample_fn = None
        self._data_loader = None
        self._param_sync_pairs = []
        self._name = name

    def set_parameter_sync(self, src_model, dst_model):
        """
        sync model parameter from src_model to dst_model

        Args
        ----
        src_model: BaseModule
            src model to sync parameters
        dst_model: BaseModule
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
        self.timers("build_sync_paramter_groups").start()
        self.model_manager.build_parameter_group()
        self.timers("build_sync_paramter_groups").stop()
        logger.info(
            f"{LOG_START} {self._name} build_sync_paramter_groups summary {self.timers.log(names=['build_sync_paramter_groups'])}")
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
        logger.info(
            f"{LOG_START} {self._name} episode summary, episode iteration {iteration + 1} {episode_str}")
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
        self.timers("chatlearn").start()
        self.timers("setup").start()
        self.setup()
        self.timers("executor_setup").start()
        for executor in self._executors:
            if executor:
                executor.setup()
        self.timers("executor_setup").stop()
        logger.info(
            f"{LOG_START} {self._name} setup executors: {self.timers.log(names=['executor_setup'])}")
        self.timers("setup").stop()
        logger.info(
            f"{LOG_START} {self._name} setup summary {self.timers.log(names=['setup'])}")
        self.logging_memory()
        self._resume_from_data_checkpoint()
        data_loader = StreamDataset.remote(self.runtime_args.stream_data_loader_type,
                                           self.runtime_args.train_micro_batch_size,
                                           self.env._padding_config,
                                           self.runtime_args.max_relay_episode,
                                           self.runtime_args.relay_episode_offset)
        logger.info(f"{LOG_START} " + get_full_proc_memory_info('Before first param sync'))
        dump_root_path = os.getenv("DEBUG_SYNC_PARAMETERS_PATH", "")
        if dump_root_path:
            if os.path.exists(dump_root_path):
                shutil.rmtree(dump_root_path)
            logger.info("dump parameters before syncnizing...")
            self.dump_parameters(os.path.join(dump_root_path, "before_sync_parameter"))
        self.timers("sync_parameters").start()
        if os.getenv("ENABLE_PARAM_SYNC_WARMUP", "false") == "true":
            self.timers("warmup_sync_parameters").start()
            self.model_manager.sync_parameters(requires_grad=False, validate=False, dryrun=True)
            self.model_manager.warmup_collective_topology()
            self.timers("warmup_sync_parameters").stop()
            logger.info(f"finish warmup_sync_parameters {self.timers.log(names=['warmup_sync_parameters'])} ")
        self.model_manager.sync_parameters(requires_grad=False, validate=self.runtime_args.validate_param_sync)
        self.timers("sync_parameters").stop()
        if self.runtime_args.enable_eval_before_training:
            self.evaluate(-1)
        if dump_root_path:
            logger.info("dump parameters after synchronizing...")
            self.dump_parameters(os.path.join(dump_root_path, "after_sync_parameter"))
            logger.info("finish dump parameters, ChatLearn will exit")
            return
        logger.info(
            f"{LOG_START} {self._name} sync_parameters summary {self.timers.log(names=['sync_parameters'])} "
            + get_full_proc_memory_info('After first param sync')
        )
        self._data_loader = data_loader
        for episode_id in range(self._start_episode, self.runtime_args.num_episode):
            if self.runtime_args.nsys:
                if episode_id == 4:
                    torch.cuda.cudart().cudaProfilerStart()
                if episode_id == 5:
                    torch.cuda.cudart().cudaProfilerStop()
            self.timers("episode").start()
            self.before_episode()
            logger.info(f"start train episode_id: {episode_id + 1}/{self.runtime_args.num_episode}")
            if self.env.timers is None:
                self.env.set_timers(self.timers)
            queue = self.env.make_experiences()
            self.timers("set_train_dataset").start()
            refs = data_loader.set_dataset.remote(queue, episode_id, self._relay_sample_fn,
                                                  self.runtime_args.sample_per_episode)
            future.wait(refs)
            if self.trainer is not None:
                # validate parameter sync in the first two episodes
                validate = self.runtime_args.validate_param_sync and episode_id < 2
                self.timers("set_train_dataset").stop()
                self.trainer.set_data_loader(data_loader)
                logger.info("set dataloader for trainer done")
                logger.info(get_full_proc_memory_info(f'Before train {episode_id}'))
                if self.trainer.timers is None:
                    self.trainer.set_timers(self.timers)
                self.trainer.train(episode_id)
                logger.info(get_full_proc_memory_info(f'After train {episode_id}'))
                logger.info(f"train episode_id: {episode_id + 1}/{self.runtime_args.num_episode} done")
                self.timers("sync_parameters").start()
                self.model_manager.sync_parameters(episode_id + 1, validate=validate)
                self.timers("sync_parameters").stop()
                logger.info(f"train episode_id: {episode_id + 1}/{self.runtime_args.num_episode} parameter sync done")
            self.after_episode()
            self.timers("episode").stop()
            self.logging_summary(episode_id)
            self.save_checkpoint(episode_id)
            self.evaluate(episode_id)

        self.timers("chatlearn").stop()
        logger.info(f"{LOG_START} {self._name} overall summary {self.timers.log(names=['chatlearn'])}")
        logger.info(f"train {self._name} done")

    def _resume_from_data_checkpoint(self):
        if self.runtime_args.data_checkpoint_path:
            data_ckpt_manager = CheckpointManager(self.models[0].replicas[0], self.runtime_args.data_checkpoint_path,
                                                  self.runtime_args.max_data_ckpt_nums,
                                                  self.runtime_args.load_data_checkpoint_iteration)
            if self.runtime_args.enable_resume_training:
                meta = data_ckpt_manager.resume_meta()
                if meta:
                    self._start_episode = meta["episode"] + 1
                    self.trainer.iteration = meta["train_iteration"]
                    if self.trainer.iteration > 0:
                        logger.info(f"ChatLearn continue train with meta {meta}")

    def dump_parameters(self, dump_path):
        for _, model in enumerate(self.models):
            replic_0 = model.replicas[0]
            if isinstance(replic_0, DistVLLMActor):
                future.wait(replic_0.vllm_engine.dump_parameters.remote(dump_path))

    def save_checkpoint(self, episode_id):
        """
        :meta private:
        """
        if self.runtime_args.save_episode_interval and \
                (episode_id + 1) % self.runtime_args.save_episode_interval == 0:
            self.timers("save_checkpoint").start()
            for model in self.trainer.models:
                refs = model.replicas[0].onload(to_onload_optimizer_states=False)
                future.wait(refs)
                refs = model.replicas[0].save_checkpoint(self.trainer.iteration)
                future.wait(refs)
                refs = model.replicas[0].offload()
                future.wait(refs)
            refs = []
            for i, model in enumerate(self.models[0].replicas):
                if isinstance(model, DistVLLMActor):
                    refs.append(model.vllm_engine.save_data_checkpoint.remote(i, self.trainer.iteration, episode_id))
                else:
                    refs.append(model.all_actors[0].save_data_checkpoint.remote(i, self.trainer.iteration, episode_id))
            future.get(refs)
            self.timers("save_checkpoint").stop()
            logger.info(f"save checkpoint episode {episode_id}, train iteration {self.trainer.iteration} done")

    def evaluate(self, episode_id):
        """
        :meta private:
        """
        if self.evaluator is not None and \
                self.runtime_args.eval_episode_interval and \
                (episode_id + 1) % self.runtime_args.eval_episode_interval == 0:
            if self.evaluator.timers is None:
                self.evaluator.set_timers(self.timers)
            logger.info("start evaluate")
            self.timers("evaluate").start()
            self.evaluator.eval(episode_id, self.trainer.iteration)
            self.timers("evaluate").stop()
            super().logging_summary(episode_id)
            logger.info(f"evaluate done {self.timers.log(names=['evaluate'])}")


class RLHFEngine(Engine):
    """RLHFEngine"""

    def __init__(self,
                 policy: BaseModule,
                 reference: BaseModule,
                 reward: BaseModule,
                 value: BaseModule,
                 policy_trainer: BaseModule,
                 value_trainer: BaseModule):
        def env_compute_flow(batch):
            policy_out = policy.forward_step(batch)
            ref_out = reference.forward_step(policy_out)
            value_out = value.forward_step(policy_out)
            reward_out = reward.forward_step(policy_out, ref_out, value_out)
            return value_out, reward_out

        def trainer_compute_flow(batch):
            policy_trainer.train_step(batch)
            value_trainer.train_step(batch)

        env = Environment(env_compute_flow)
        trainer = Trainer(trainer_compute_flow)
        super().__init__(env, trainer, name='rlhf')
        self.set_parameter_sync(policy_trainer, policy)
        self.set_parameter_sync(value_trainer, value)


class OnlineDPOEngine(Engine):
    """Online DPO Engine."""
    def __init__(self,
                 policy: BaseModule,
                 reference: BaseModule,
                 reward: BaseModule,
                 policy_trainer: BaseModule):
        def env_compute_flow(batch):
            policy_out = policy.forward_step(batch)
            ref_out = reference.forward_step(policy_out)
            reward_out = reward.forward_step(policy_out, ref_out)
            return reward_out

        def trainer_compute_flow(batch):
            policy_trainer.train_step(batch)

        env = Environment(env_compute_flow)
        trainer = Trainer(trainer_compute_flow)
        super().__init__(env, trainer, name='online_dpo')
        self.set_parameter_sync(policy_trainer, policy)


class DPOEngine(Engine):
    """DPO Engine."""
    def __init__(self,
                 reference: BaseModule,
                 policy_trainer: BaseModule):
        def env_compute_flow(batch):
            ref_out = reference.forward_step(batch)
            return ref_out

        def trainer_compute_flow(batch):
            policy_trainer.train_step(batch)

        env = Environment(env_compute_flow)
        trainer = Trainer(trainer_compute_flow)
        super().__init__(env, trainer, name='dpo')


class GRPOEngine(Engine):
    """GRPO Engine."""
    def __init__(self,
                 policy: BaseModule,
                 reference: BaseModule,
                 reward: BaseModule,
                 policy_trainer: BaseModule):
        def env_compute_flow(batch):
            policy_out = policy.forward_step(batch)
            ref_out = reference.forward_step(policy_out)
            reward_out = reward.forward_step(policy_out, ref_out)
            return reward_out

        def trainer_compute_flow(batch):
            policy_trainer.train_step(batch)

        env = Environment(env_compute_flow)
        trainer = Trainer(trainer_compute_flow)
        super().__init__(env, trainer, name='grpo')
        self.set_parameter_sync(policy_trainer, policy)


class GRPOMathEngine(Engine):
    """GRPO Engine with math reward"""
    def __init__(self,
                 policy,
                 reference,
                 reward,
                 reward1,
                 ppo_policy):

        def env_compute_flow(batch):
            policy_out = policy.forward_step(batch)
            ref_out = reference.forward_step(policy_out)
            reward_out = reward.forward_step(policy_out, ref_out)
            reward_out1 = reward1.forward_step(batch, policy_out)
            return reward_out, reward_out1

        def trainer_compute_flow(batch):
            ppo_policy.train_step(batch)

        def evaluator_flow(batch):
            policy_out = policy.eval_forward(batch)
            reward_out = reward.eval_forward(policy_out)
            reward_out1 = reward1.eval_forward(policy_out)
            return reward_out, reward_out1

        env = Environment(env_compute_flow)
        trainer = Trainer(trainer_compute_flow)
        evaluator = Evaluator(evaluator_flow)
        super().__init__(env, trainer, evaluator, name='grpo_math')
        self.set_parameter_sync(ppo_policy, policy)


class EvalEngine(Engine):
    """Evaluation Engine"""

    def __init__(self, eval_flow=None, evaluator=None):
        if evaluator is None:
            evaluator = Evaluator(eval_flow)
        super().__init__(evaluator=evaluator)

    def setup(self):
        super().setup()
        self.evaluator.set_dataset(self._dataset)
        self.evaluator.set_timers(self.timers)
        self.evaluator.set_post_process_func(self._post_process_func)

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

    def set_post_process_func(self, post_process_func):
        """
        Set post process function.

        Args
        ----
        post_process_func
            This function accept two arguments.
            1. results: a list of evaluation results
            2. eval_info: a dict meta that contains "train_iteration" and "episode_iteration"
        """
        self._post_process_func = post_process_func
        return self

    def eval(self, cur_iter=None, train_iteration=None):
        """
        Start evaluating.
        """
        self.setup()
        self.evaluator.setup()
        self.timers("episode").start()
        results = self.evaluator.eval(
            cur_iter=cur_iter, train_iteration=train_iteration)
        self.timers("episode").stop()
        return results
