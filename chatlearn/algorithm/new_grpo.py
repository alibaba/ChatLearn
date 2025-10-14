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
"""grpo algorithm"""

from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Any
import traceback
import itertools
import functools
import time
from contextlib import contextmanager

import ray

import chatlearn

from chatlearn.algorithm.grpo_utils.advantage_compute import AdvantageComputer
from chatlearn.algorithm.grpo_utils.policy_trainer import PolicyTrainer
from chatlearn.algorithm.grpo_utils.partial_rollout_manager import PartialRolloutManager
from chatlearn.algorithm.grpo_utils.packing_utils import regroup_data_packing_simple
from chatlearn.algorithm.base_algo import BaseAlgorithm

from chatlearn.models.vllm_module import VLLMModule
from chatlearn.models.sglang_module import SGLangModule, AsyncSGLangModule
from chatlearn.models.agent.agent_module import AgentModule
from chatlearn.models.reward.rule_reward import RuleReward
from chatlearn.models.agent.rollout_manager import RolloutManager

from chatlearn.schedule.model_manager import ModelManager
from chatlearn.schedule.resource_manager import ResourceManager
from chatlearn.schedule.metric_manager import MetricManager

try:
    from chatlearn.utils.megatron_utils import update_cfg
    from chatlearn.algorithm.grpo_utils.megatron_policy_trainer import \
        MegatronPolicyTrainer
except Exception:
    traceback.print_exc()
    print("please set megatron path for running megatron backend")

from chatlearn.utils.logger import logger
from chatlearn.utils.utils import \
    get_full_proc_memory_info, map_reduce_metrics, even_slice, rearrange_zigzag

from chatlearn.data.prompt_dataset import PromptPipeline
from chatlearn.data.data import read_data_path_list, RLHFDataLoader_Simple
from chatlearn.data.sampler import MultiDatasetSampler_Simple

@contextmanager
def timer(log_name, timer_summary):
    start = time.time()
    yield
    end = time.time()
    timer_summary[log_name] = end - start

def idle_collate(batch):
    return batch

class GrpoAlgorithm(BaseAlgorithm):
    """GrpoAlgorithm"""

    def __init__(self, cfg) -> None:
        if cfg.runtime_args.train_backend == "megatron":
            cfg = update_cfg(cfg)
        self.cfg = cfg
        chatlearn.init(self.cfg)
        self.models = self.create_models()
        self.eval_dataiter = self.get_data_iter(is_eval=True)
        self.train_dataiter = self.get_data_iter(is_eval=False)
        self.adv_computer = AdvantageComputer(self.cfg.runtime_args.num_inference_per_prompt)
        self._metrics_gather = {}
        self._timing_summary = defaultdict(float)

    def create_models(self):
        # FSDP/Megatron Models
        if self.cfg.runtime_args.train_backend == "fsdp":
            self.policy_trainer = PolicyTrainer("policy_trainer")
            self.ref_policy = PolicyTrainer("ref_policy")
        elif self.cfg.runtime_args.train_backend == "megatron":
            self.policy_trainer = MegatronPolicyTrainer("policy_trainer")
            self.ref_policy = MegatronPolicyTrainer("ref_policy")

        # Rollout Models
        if self.cfg.runtime_args.task_type == "chat":
            if self.cfg.runtime_args.rollout_backend == "vllm":
                self.policy = VLLMModule("policy")
            elif self.cfg.runtime_args.rollout_backend == "sglang":
                rollout_cls = SGLangModule if self.cfg.models.policy.is_sync_mode else AsyncSGLangModule
                self.policy = rollout_cls("policy")
        elif self.cfg.runtime_args.task_type == "agent":
            assert not self.cfg.models.policy.is_sync_mode and self.cfg.runtime_args.rollout_backend == "sglang", \
                "agent task only support async sglang engine"
            assert self.cfg.runtime_args.use_rollout_manager, "agent task must set use_rollout_manager=True"
            self.policy = AgentModule("policy")

        # Reward Models
        self.reward = RuleReward("reward")

    def get_data_iter(self, is_eval):
        if is_eval:
            data_path_list = [
                item.strip() for item in self.cfg.runtime_args.eval_data_path.split(",")
            ]
        else:
            data_path_list = [
                item.strip() for item in self.cfg.runtime_args.data_path.split(",")
            ]
        data = read_data_path_list(data_path_list)
        tokeniser_path = self.cfg.models.policy.load
        max_prompt_tokens_length = self.cfg.models.policy.max_prompt_tokens_length
        dataset = PromptPipeline(
            data,
            max_prompt_tokens_length=max_prompt_tokens_length,
            tokenizer=tokeniser_path,
            enable_thinking=self.cfg.models.policy.enable_thinking,
            raw_chat=False,
        )
        datasets = []
        datasets.append(dataset)
        dataset_size = [len(dataset) for dataset in datasets]
        if is_eval:
            sampler = MultiDatasetSampler_Simple(
                dataset_sizes = dataset_size,
                sample_per_episode=sum(dataset_size),
                shuffle=False,
                is_eval=True,
            )
        else:
            sampler = MultiDatasetSampler_Simple(
                dataset_sizes = dataset_size,
                sample_per_episode=self.cfg.runtime_args.sample_per_episode,
                num_inference_per_prompt=self.cfg.runtime_args.num_inference_per_prompt,
                is_eval=False,
                shuffle=self.cfg.runtime_args.data_shuffle,
                data_rerank=self.cfg.runtime_args.data_rerank
            )

        dataloader = RLHFDataLoader_Simple(
            datasets,
            sampler,
            collate_fn=idle_collate,
        )
        dataiter = iter(dataloader)
        return dataiter

    def init_resources(self):
        # Create metrics
        self.metric_manager = MetricManager(self.cfg)

        # Create ray actors
        resource_manager = ResourceManager()
        self.model_manager = ModelManager(resource_manager, self.cfg)

        # TODO: too ugly, try clean
        self.model_manager.create_dist_models([self.policy_trainer, self.ref_policy, self.policy, self.reward])
        self.policy_trainer, self.ref_policy, self.policy, self.reward = self.model_manager.dist_models

        # create sync pair
        self.model_manager.set_parameter_sync(self.policy_trainer, self.policy)

        refs = []
        self.policy.init()
        for replica in self.policy.replicas:
            if self.cfg.runtime_args.rollout_backend == 'vllm':
                refs.extend(replica.setup_engine(replica.all_actors))
            else:
                refs.extend(replica.setup_engine())
        ray.get(refs)
        self.policy.offload()

        # init train actor and other rollout actors
        self.num_dp_ranks = {}
        self.policy_trainer.init()
        self.ref_policy.init()
        self.reward.init()
        for model in [self.policy_trainer, self.ref_policy, self.policy, self.reward]:
            model.model_setup()
            model.group_dist_actors_by_dp_rank()

        # Build parameter sync group and do first sync
        self.model_manager.build_parameter_group()
        self.model_manager.sync_parameters()

    def create_data_iterator(self, data_list, num_batch):
        slice_index = even_slice(len(data_list), num_batch)
        batches = []
        for start, end in zip(slice_index[:-1], slice_index[1:]):
            batches.append(data_list[start:end])
        return iter(batches)

    def eval(self):
        # Get eval data
        data = next(self.eval_dataiter)
        data = self.policy.forward_step(data, is_eval=True)
        eval_rollout_metric_list = self.gather_metrics(self.policy)
        data = self.reward.forward_step(list(itertools.chain.from_iterable(data)))
        eval_metric_list = self.gather_metrics(self.reward)
        return True

    def train(self):
        data = next(self.train_dataiter)
        # Rollout
        with timer('policy|forward_step', self._timing_summary):
            self.policy.onload()
            data = self.policy.forward_step(data)
            data = list(itertools.chain.from_iterable(data))
            total_valid_tokens = sum(d['response_token_length'] + d['prompt_token_length'] for d in data)
            self.policy.offload()
            self.gather_metrics(self.policy)

        # Reward/Adv
        with timer('reward|forward_step', self._timing_summary):
            data = self.reward.forward_step(data)
            data = list(itertools.chain.from_iterable(data))
            self.gather_metrics(self.reward)
        
        data = self.adv_computer(data)

        # calculate total_minibsz needed. num_microbsz should be divided by total_minibsz
        dp_size = self.policy_trainer.data_parallel_size
        num_minibsz = self.cfg.runtime_args.sample_per_episode // self.cfg.runtime_args.train_global_batch_size
        total_minibsz = num_minibsz * dp_size
        # Split list of sample to list of microbatches
        packed_data = regroup_data_packing_simple(data, total_minibsz, self.cfg.models.policy_trainer.max_token_in_packing, 0)
        packed_data = rearrange_zigzag(packed_data, total_minibsz)

        # ref_logprobs
        with timer('ref_policy|forward_step', self._timing_summary):
            self.ref_policy.onload()
            data = self.ref_policy.forward_step(packed_data)
            data = list(itertools.chain.from_iterable(data))
            self.ref_policy.offload()

        # old_logprobs
        with timer('policy_trainer|forward_step', self._timing_summary):
            self.policy_trainer.onload()
            data = self.policy_trainer.forward_step(data)
            data = list(itertools.chain.from_iterable(data))

        # train
        minibsz_iter = self.create_data_iterator(data, num_minibsz)
        with timer('policy_trainer|train_step', self._timing_summary):
            for minibsz_idx in range(num_minibsz):
                data = self.policy_trainer.train_step(next(minibsz_iter))
            self.policy_trainer.offload()
            self.gather_metrics(self.policy_trainer)
        return total_valid_tokens

    def gather_metrics(self, model):
        metric_list = model.get_and_clear_metrics()
        # Metrics Aggr
        prefix = metric_list[0][0]
        list_of_dict = [metric_tuple[1] for metric_tuple in metric_list]
        keys = list_of_dict[0].keys()
        means = {k: sum(d[k] for d in list_of_dict) / len(list_of_dict) for k in keys}
        self._metrics_gather[prefix] = means

    def run(self) -> None:
        start_time = time.time()
        self.init_resources()
        print(f"debugyy init resources time: {time.time() - start_time}")

        # Do eval first
        if self.cfg.runtime_args.enable_eval_before_training:
            self.eval()

        # Start training
        episode_id = 0
        while episode_id < self.cfg.runtime_args.num_episode:
            episode_id += 1
            with timer('episode_e2e', self._timing_summary):
                total_tokens = self.train()
                self.model_manager.sync_parameters()

            self._timing_summary['tps'] = total_tokens / self._timing_summary['episode_e2e']

            for k, v in self._metrics_gather.items():
                self.metric_manager.log(k, episode_id, v)

            self.metric_manager.log('train_time_summary', episode_id, self._timing_summary)
            self._timing_summary.clear()
            self._metrics_gather = {}
            if episode_id % self.cfg.runtime_args.eval_episode_interval == 0:
                with timer('eval_e2e', self._timing_summary):
                    self.eval()

                for k, v in self._metrics_gather.items():
                    self.metric_manager.log(f"eval_{k}", episode_id, v)

                self.metric_manager.log('eval_time_summary', episode_id, self._timing_summary)
                self._timing_summary.clear()
                self._metrics_gather = {}

        # put data in engine._all_datasets
    def validate(self):
        self.cfg.validate()
