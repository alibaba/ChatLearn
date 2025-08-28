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
import time
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from ray.actor import ActorHandle


from chatlearn.checkpoint.checkpoint_manager import CheckpointManager
from chatlearn.data.data import StreamDataset
from chatlearn.models.base_module import BaseModule
from chatlearn.runtime.dist_actor import DistVLLMActor, DistSGLangActor
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.trainer import Trainer
from chatlearn.schedule.model_manager import ModelManager
from chatlearn.schedule.resource_manager import ResourceManager
from chatlearn.schedule.metric_manager import MetricManager
from chatlearn.utils import future
from chatlearn.utils.constant import LOG_START
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.utils.utils import get_full_proc_memory_info
from chatlearn.utils.timer import Timers
from chatlearn.utils.utils import map_reduce_metrics


class BaseEngine:
    """Base Engine"""

    def __init__(self, *models):
        self._models = models
        self.global_args = get_args()
        self.runtime_args = self.global_args.runtime_args
        self._timers = Timers()
        self.writer_dict = {}

    def set_timers(self, _timers):
        self._timers = _timers

    @property
    def timers(self):
        return self._timers

    def timer_summary(self):
        """
        :meta private:
        """
        return self._timers.log(reset=False, return_dict=True)

    def _create_remote_models(self):
        resource_manager = ResourceManager(self._models)
        self.model_manager = ModelManager(self._models, resource_manager, self.global_args)
        self.model_manager.remote()
        self.remote_models = self.model_manager.dist_models
        self.named_models = {model.name: model for model in self.remote_models}

    def _create_metric_manager(self):
        self.metric_manager = MetricManager(self.global_args)

    def setup(self):
        """
        1. create MetricManager、ResourceManager、ModelManager
        2. create all remote models
        3. init remote models
        4. model_setup for remote models
        """
        logger.info(f"{LOG_START} setup, start to create_remote_models")
        self._create_metric_manager()
        t1 = time.time()
        self._create_remote_models()
        t2 = time.time()
        logger.info(f"{LOG_START} setup, finished to create_remote_models(s):{(t2-t1)}")
        # for ease to access model by self.{model_name}
        for model in self.remote_models:
            setattr(self, model.name, model)

        # include compile in init, compile dependencies need to be called serially
        logger.info(get_full_proc_memory_info(f"{LOG_START} Before model init"))
        for model in self.remote_models:
            model.init()
        logger.info(get_full_proc_memory_info(f"{LOG_START} After model init"))
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
                logger.info(f"{LOG_START} start setup and validate {model.name}")
                future.wait(model.model_setup())
                future.wait(model.validate())
                logger.info(f"{LOG_START} done setup and validate {model.name}")
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

    def reduce_timer(self, timer_data: List[List[Tuple[str, dict]]]) -> Dict:
        """timer_data: (replica_num, actor_num)
        """
        # flatten
        flattened_timer_data = [item[1] for sublist in timer_data for item in sublist]
        merged_timer = defaultdict(list)
        reduce_timer = {}
        for timer_data_item in flattened_timer_data:
            for k, v in timer_data_item.items():
                merged_timer[k].append(v)

        for k, v in merged_timer.items():
            if k in ['forward_step', 'train_step', 'eval_forward']:
                reduce_timer.update(
                    {
                        f"{k}/avg(s)": sum(v) / len(v),
                        f"{k}/max(s)": max(v),
                        f"{k}/min(s)": min(v)
                    }
                )
            else:
                reduce_timer.update(
                    {
                        f"{k}(s)": sum(v) / len(v),
                    }
                )

        return reduce_timer

    def logging_summary(self, iteration=-1):
        _, e2e_time_dict = self.timer_summary()

        summaries = []
        logger.info(f"{LOG_START} episode iteration {iteration + 1} time summary.")
        for model in self.remote_models:
            timer_data = future.get(model.timer_summary(e2e_cost=e2e_time_dict.get(model.name, None)))
            reduce_timer_data = self.reduce_timer(timer_data)
            summaries.append(reduce_timer_data)

        for key, value in e2e_time_dict.items():
            e2e_time_dict[key] = {'e2e': value}

        for model, summary in zip(self.remote_models, summaries):
            if model.name not in e2e_time_dict:
                e2e_time_dict[model.name] = {}
            e2e_time_dict[model.name].update(summary)

        self.logging_memory()
        return e2e_time_dict

    def stop(self):
        self.metric_manager.stop()
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
        self._all_datasets = None
        self._post_process = None
        self._drop_last = False
        self._wrap_data = True
        self._replay_sample_manager = None
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
        return self

    def _create_remote_models(self):
        """
        :meta private:
        """
        logger.info(f"{LOG_START} create_remote_models, start to create resource_manager")
        t1 = time.time()
        resource_manager = ResourceManager(self._models)
        t2 = time.time()
        logger.info(f"{LOG_START} create_remote_models, finished to create resource_manager(s):{(t2-t1)}")
        self.model_manager = ModelManager(self._models, resource_manager, self.global_args)
        for src_model, dst_model in self._param_sync_pairs:
            self.model_manager.set_parameter_sync(src_model, dst_model)
        self.model_manager.remote()
        t3 = time.time()
        logger.info(f"{LOG_START} create_remote_models, finished to set_parameter_sync(s):{(t3-t2)}")
        self.remote_models = self.model_manager.dist_models
        self.named_models = {model.name: model for model in self.remote_models}
        t4 = time.time()
        logger.info(f"{LOG_START} create_remote_models, finished to get named_models(s):{(t4-t3)}")

    def setup(self):
        """
        1. BaseEngine.setup()
        2. set the Initialized DistModel to executor
        3. set dataset to env
        4. build_parameter_group for parameter sync
        """
        super().setup()
        self._executors = [self.env, self.trainer, self.evaluator]
        for executor in self._executors:
            if executor:
                executor.update_models(self.remote_models)
        if self.env:
            self.env.set_multiple_datasets(self._all_datasets)
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
        dataset : list[str]
            a list of prompt string
        """
        assert isinstance(dataset, list), (
            f"expect datasets to be a list, got {type(dataset)}"
        )
        assert not isinstance(dataset[0], list), (
            "expect only one dataset to be set, if you want to use more "
            "than one dataset, please try `set_multiple_datasets`"
        )
        self._all_datasets = [dataset]
        return self

    def set_multiple_datasets(self, all_datasets):
        """
        Set multiple prompt datasets.

        Args
        ----
        all_datasets : list[list[str]]
            a list of lists of prompt string
        """
        # sanity check
        assert len(all_datasets) >= 1, (
            f"expect at least one dataset, got {len(all_datasets)} datasets."
        )
        assert isinstance(all_datasets, list), (
            f"expect datasets to be a list, got {type(all_datasets)}"
        )
        for dataset in all_datasets:
            assert isinstance(dataset, list), (
                f"expect each dataset to be a list of prompts, got {type(dataset)}"
            )

        self._all_datasets = all_datasets
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
        ## 1. model e2e time
        e2e_time_dict = super().logging_summary(iteration)
        # flatten time to name/<e2e or forward_step or eval_step and so on>
        model_time_dict = {}
        for model in self.remote_models:
            model_e2e_time_dict = e2e_time_dict.get(model.name, {})
            for key, value in model_e2e_time_dict.items():
                model_time_dict[f"{model.name}/{key}"] = value

        ## 2. episode time
        timer_names = ['sync_parameters']
        # timer_names before episode looping
        if iteration == -1 and self.evaluator and self.runtime_args.enable_eval_before_training:
            timer_names.append('evaluate')
        # timer_names in episode looping
        elif iteration >= 0:
            timer_names.extend(['episode', 'train', 'environment'])
            if self.runtime_args.save_episode_interval and \
                    (iteration + 1) % self.runtime_args.save_episode_interval == 0:
                timer_names.append('save_checkpoint')
            if self.evaluator is not None and \
                    self.runtime_args.eval_episode_interval and \
                    (iteration + 1) % self.runtime_args.eval_episode_interval == 0:
                timer_names.append('evaluate')

        episode_str, episode_metrics = self.timers.log(names=timer_names, return_dict=True)

        log_str = f"{LOG_START} {self._name} episode summary, episode {iteration + 1} {episode_str}"
        logger.info(log_str)

        ## 3. log model e2e time and episode time
        episode_metrics.update(model_time_dict)
        self.metric_manager.log("engine/timer_summary", iteration + 1, episode_metrics)

        ## 4. log before episode looping
        if iteration == -1:
            if self.evaluator and self.runtime_args.enable_eval_before_training:
                prefix, evaluate_metrics = self.evaluator.get_and_clear_metrics()
                self.metric_manager.log(prefix, iteration + 1, evaluate_metrics)
            return

        ## 5. log in episode looping
        # Train metrics
        for model in self.remote_models:
            all_metric_tuples = future.get(model.get_and_clear_metrics())
            if isinstance(all_metric_tuples[0], list):
                all_metric_tuples_flaten = []
                for item in all_metric_tuples:
                    all_metric_tuples_flaten += item
                all_metric_tuples = all_metric_tuples_flaten
            prefix = all_metric_tuples[0][0]
            last_rank_metrics = [metric_tuple[1] for metric_tuple in all_metric_tuples]
            model_metrics = map_reduce_metrics(last_rank_metrics)
            self.metric_manager.log(prefix, iteration + 1, model_metrics)
        # Reward metrics
        if self._data_loader:
            prefix, train_reward_metrics = future.get(self._data_loader.get_and_clear_metrics.remote())
            self.metric_manager.log(prefix, iteration + 1, train_reward_metrics)
        # Evaluate metrics
        if self.evaluator:
            prefix, evaluate_metrics = self.evaluator.get_and_clear_metrics()
            self.metric_manager.log(prefix, iteration + 1, evaluate_metrics)


    def set_replay_sample_manager(self, replay_sample_manager):
        """
        Set custom replay_sample_manager.

        Args
        ----
            replay_sample_manager: inputs List[EpisodeReplayBuffer], return a list of dict.
        """
        self._replay_sample_manager = replay_sample_manager

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

        data_loader: ActorHandle = StreamDataset.remote(
            self.runtime_args.stream_data_loader_type,
            self.runtime_args.sample_per_episode // self.runtime_args.train_global_batch_size,
            self.runtime_args.train_micro_batch_size,
            self.runtime_args.max_replay_episode,
            self.runtime_args.replay_episode_offset
        )

        logger.info(f"{LOG_START} " + get_full_proc_memory_info('Before first param sync'))
        dump_root_path = os.getenv("DEBUG_SYNC_PARAMETERS_PATH", "")
        if dump_root_path:
            if os.path.exists(dump_root_path):
                shutil.rmtree(dump_root_path)
            logger.info(f"{LOG_START} dump parameters before syncnizing...")
            self.dump_parameters(os.path.join(dump_root_path, "before_sync_parameter"))
        self.timers("sync_parameters").start()
        self.model_manager.sync_parameters()
        self.timers("sync_parameters").stop()
        if self.runtime_args.enable_eval_before_training:
            self.evaluate(-1)
        if dump_root_path:
            logger.info(f"{LOG_START} dump parameters after synchronizing...")
            self.dump_parameters(os.path.join(dump_root_path, "after_sync_parameter"))
            logger.info(f"{LOG_START} finish dump parameters, ChatLearn will exit")
            return
        logger.info(get_full_proc_memory_info('After first param sync'))
        self.logging_summary(-1)

        self._data_loader = data_loader
        for episode_id in range(self._start_episode, self.runtime_args.num_episode):
            if self.runtime_args.nsys:
                if episode_id == 4:
                    torch.cuda.cudart().cudaProfilerStart()
                if episode_id == 5:
                    torch.cuda.cudart().cudaProfilerStop()
            self.timers("episode").start()
            self.before_episode()
            logger.info(f"{LOG_START} start train episode_id: {episode_id + 1}/{self.runtime_args.num_episode}")
            if self.env.timers is None:
                self.env.set_timers(self.timers)
            queue = []
            if os.getenv("SKIP_GENERATION", None) is None:
                logger.info(f"{LOG_START} start to make experience: {episode_id + 1}/{self.runtime_args.num_episode}")
                self.timers("environment").start()
                queue = self.env.make_experiences()
                self.timers("environment").stop()
                logger.info(f"{LOG_START} complete to make experience: {episode_id + 1}/{self.runtime_args.num_episode}")
                self.timers("set_train_dataset").start()
            else:
                logger.info(f"{LOG_START} Skip generation phase for episode_id: {episode_id + 1}/{self.runtime_args.num_episode}")
            refs = data_loader.set_dataset.remote(queue, episode_id, self._replay_sample_manager,
                                                  self.runtime_args.sample_per_episode)
            future.wait(refs, return_output=True)
            if self.trainer is not None:
                self.timers("set_train_dataset").stop()
                self.trainer.set_data_loader(data_loader)
                logger.info(f"{LOG_START} set dataloader for trainer done")
                logger.info(get_full_proc_memory_info(f"{LOG_START} Before train {episode_id}"))
                if self.trainer.timers is None:
                    self.trainer.set_timers(self.timers)

                self.timers("train").start()
                self.trainer.train(episode_id)
                self.timers("train").stop()

                logger.info(get_full_proc_memory_info(f"{LOG_START} After train {episode_id}"))

                self.save_checkpoint(episode_id)

                logger.info(f"{LOG_START} save episode_id: {episode_id + 1}/{self.runtime_args.num_episode} done")
                self.timers("sync_parameters").start()
                self.model_manager.sync_parameters(episode_id + 1)
                self.timers("sync_parameters").stop()
                logger.info(f"{LOG_START} train episode_id: {episode_id + 1}/{self.runtime_args.num_episode} parameter sync done")
            logger.info(f"{LOG_START} train episode_id: {episode_id + 1}/{self.runtime_args.num_episode} done")
            self.timers("episode").stop()
            self.evaluate(episode_id)
            self.after_episode()
            self.logging_summary(episode_id)

        self.timers("chatlearn").stop()
        logger.info(f"{LOG_START} {self._name} overall summary {self.timers.log(names=['chatlearn'])}")
        logger.info(f"{LOG_START} train {self._name} done")

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
                        logger.info(f"{LOG_START} continue train with meta {meta}")

    def dump_parameters(self, dump_path):
        for _, model in enumerate(self.models):
            replic_0 = model.replicas[0]
            if isinstance(replic_0, DistVLLMActor):
                future.wait(replic_0.engine.dump_parameters.remote(dump_path))

    def save_checkpoint(self, episode_id):
        """
        :meta private:
        """
        if self.runtime_args.save_episode_interval and \
                (episode_id + 1) % self.runtime_args.save_episode_interval == 0:
            self.timers("save_checkpoint").start()
            for model in self.trainer.models:
                refs = model.replicas[0].onload(to_onload_optimizer_states=False)
                future.wait(refs, return_output=True)
                refs = model.replicas[0].save_checkpoint(self.trainer.iteration)
                future.wait(refs, return_output=True)
                refs = model.replicas[0].offload()
                future.wait(refs, return_output=True)
            refs = []
            for i, model in enumerate(self.models[0].replicas):
                if isinstance(model, (DistVLLMActor, DistSGLangActor)):
                    refs.append(model.engine.save_data_checkpoint.remote(i, self.trainer.iteration, episode_id))
                else:
                    refs.append(model.all_actors[0].save_data_checkpoint.remote(i, self.trainer.iteration, episode_id))
            future.get(refs)
            self.timers("save_checkpoint").stop()
            logger.info(f"{LOG_START} save checkpoint episode {episode_id}, train iteration {self.trainer.iteration} done")

    def evaluate(self, episode_id):
        """
        :meta private:
        """
        if self.evaluator is not None and \
                self.runtime_args.eval_episode_interval and \
                (episode_id + 1) % self.runtime_args.eval_episode_interval == 0:
            if self.evaluator.timers is None:
                self.evaluator.set_timers(self.timers)
            logger.info(f"{LOG_START} start evaluate")
            self.timers("evaluate").start()
            self.evaluator.eval(episode_id, self.trainer.iteration)
            self.timers("evaluate").stop()
            logger.info(f"{LOG_START} evaluate done")

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
        