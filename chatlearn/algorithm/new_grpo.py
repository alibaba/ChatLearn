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

from algorithm.base_algo import BaseAlgorithm

import chatlearn
from chatlearn import Engine
from chatlearn.configs import (
    BaseConfig,
    RewardConfig,
    RolloutManagerConfig,
    PolicyConfig,
    RuntimeConfig,
    RuntimeEnvConfig,
    PartialRolloutManagerConfig
)
from chatlearn.configs.fsdp_config import FSDPPolicyTrainerConfig, FSDPRefPolicyConfig

from chatlearn.algorithm.grpo_utils.advantage_compute import AdvantageComputer
from chatlearn.algorithm.grpo_utils.policy_trainer import PolicyTrainer
from chatlearn.models.vllm_module import VLLMModule
from chatlearn.models.sglang_module import SGLangModule, AsyncSGLangModule
from chatlearn.models.torch_module import TorchModule
from chatlearn.models.agent.agent_module import AgentModule
from chatlearn.algorithm.grpo_utils.partial_rollout_manager import PartialRolloutManager
from chatlearn.data.data import read_data_path_list
from chatlearn.models.reward.rule_reward import RuleReward
from chatlearn.models.agent.rollout_manager import RolloutManager
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.evaluator import Evaluator
from chatlearn.runtime.trainer import Trainer

try:
    from chatlearn.utils.megatron_utils import update_cfg
    from chatlearn.algorithm.grpo_utils.megatron_policy_trainer import \
        MegatronPolicyTrainer
    from chatlearn.configs.megatron_config import (
        MegatronPolicyTrainerConfig,
        MegatronRefPolicyConfig
    )
except Exception:
    traceback.print_exc()
    print("please set megatron path for running megatron backend")


@dataclass
class GrpoModelConfig(BaseConfig):
    """GrpoModelConfig"""
    policy: PolicyConfig = field(
        default_factory=PolicyConfig, metadata={"help": "Policy config."}
    )
    reward: RewardConfig = field(
        default_factory=RewardConfig, metadata={"help": "Reward config."}
    )
    rollout_manager: RolloutManagerConfig = field(
        default_factory=RolloutManagerConfig, metadata={"help": "rollout manager config."}
    )
    partial_rollout_manager: PartialRolloutManagerConfig = field(
        default=PartialRolloutManagerConfig, metadata={"help": "partial Rollout manager config. Only useful when partial_rollout is enabled"}
    )
    ref_policy: Any = field(
        default=None,
        metadata={
            "help": "Reference policy config. One of RefPolicyConfig or MegatronRefPolicyConfig."
        },
    )
    policy_trainer: Any = field(
        default=None,
        metadata={
            "help": "Policy trainer config. One of PolicyTrainerConfig or MegatronPolicyTrainerConfig."
        },
    )


@dataclass
class GrpoConfig(BaseConfig):
    """GrpoConfig"""

    env_args: RuntimeEnvConfig = field(
        default_factory=RuntimeEnvConfig,
        metadata={"help": "Runtime environment config."},
    )
    runtime_args: RuntimeConfig = field(
        default_factory=RuntimeConfig, metadata={"help": "Runtime config."}
    )
    models: GrpoModelConfig = field(
        default_factory=GrpoModelConfig, metadata={"help": "Grpo model config."}
    )

    def __post_init__(self):
        def convert_to_dataclass(cls, data):
            if isinstance(data, dict):
                field_types = {f.name: f.type for f in fields(cls)}
                converted = {}
                for k, v in data.items():
                    if k in field_types and isinstance(v, dict):
                        converted[k] = convert_to_dataclass(field_types[k], v)
                    else:
                        converted[k] = v
                return cls(**converted)
            return data

        train_backend = self.runtime_args.train_backend
        if train_backend == "fsdp":
            refpolicy_cls, policytrainer_cls = FSDPRefPolicyConfig, FSDPPolicyTrainerConfig
        elif train_backend == "megatron":
            refpolicy_cls, policytrainer_cls = (
                MegatronRefPolicyConfig,
                MegatronPolicyTrainerConfig,
            )
        else:
            raise Exception(f"not support train backend: {train_backend}")
        self.models.ref_policy = convert_to_dataclass(
            refpolicy_cls, self.models.ref_policy
        )
        self.models.policy_trainer = convert_to_dataclass(
            policytrainer_cls, self.models.policy_trainer
        )

    def _validate_impl(self):
        sample_per_episode = self.runtime_args.sample_per_episode
        policy = self.models.policy
        assert sample_per_episode % policy.num_inference_per_prompt == 0, \
        "runtime_args.sample_per_episode must be divisible by models.policy.num_inference_per_prompt"
        assert sample_per_episode % policy.replica_dp_size == 0, (
            "runtime_args.sample_per_episode must be divisible by dp_size of policy model"
        )
        models = {
            'policy_trainer': self.models.policy_trainer,
            'ref_policy': self.models.ref_policy
        }
        for name, conf in models.items():
            # NOTE: sample_per_episode should be divided by total DP
            assert sample_per_episode % (conf.num_replica * conf.replica_dp_size) == 0, (
                f"runtime_args.sample_per_episode of {name} ({self.runtime_args.sample_per_episode}) must be divisible "
                f"by models.{name}.num_replica ({conf.num_replica}) times models.{name}.replica_dp_size ({conf.replica_dp_size})."
            )
            if conf.trainable:
                # NOTE: train_global_batch_size should be divided by total DP if trainable
                assert self.runtime_args.train_global_batch_size % (conf.num_replica * conf.replica_dp_size) == 0, (
                    f"runtime_args.train_global_batch_size ({self.runtime_args.train_global_batch_size}) must be divisible by "
                    f"models.{name}.num_replica ({conf.num_replica}) times models.{name}.replica_dp_size ({conf.replica_dp_size})."
                )

            if not conf.packing:
                sample_per_dp_rank = sample_per_episode // (conf.num_replica * conf.replica_dp_size)
                assert sample_per_dp_rank % conf.generation_batch_size == 0, (
                    f"sample_per_dp_rank of {name} ({sample_per_dp_rank}) must be divisible by "
                    f"models.{name}.generation_batch_size ({conf.generation_batch_size})."
                )

                if conf.trainable:
                    train_global_batch_size_per_dp_rank = (
                        self.runtime_args.train_global_batch_size // (conf.num_replica * conf.replica_dp_size)
                    )
                    assert train_global_batch_size_per_dp_rank % self.runtime_args.train_micro_batch_size == 0, (
                        f"train_global_batch_size_per_dp_rank of {name} ({sample_per_dp_rank}) must be divisible by "
                        f"runtime_args.train_micro_batch_size ({self.runtime_args.train_micro_batch_size})."
                    )

from chatlearn.schedule.model_manager import ModelManager
from chatlearn.schedule.resource_manager import ResourceManager
from chatlearn.schedule.metric_manager import MetricManager
from chatlearn.utils import future
from chatlearn.utils.constant import LOG_START
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.utils.utils import get_full_proc_memory_info
from chatlearn.utils.timer import Timers, timing
from chatlearn.utils.utils import map_reduce_metrics, even_slice, rearrange_zigzag
import ray
import itertools
from chatlearn.data.prompt_dataset import PromptPipeline
from chatlearn.data.data import RLHFDataLoader_Simple
from chatlearn.data.sampler import MultiDatasetSampler_Simple
from chatlearn.algorithm.grpo_utils.packing_utils import (
    regroup_data_packing_simple
)
import time
import functools
def timeit():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            model_name = kwargs['model_name']
            func_name = kwargs['func_name']
            start = time.time()
            ret = func(self, *args, **kwargs)
            time_cost = time.time() - start
            self._timing_summary[f"{model_name}|{func_name}"] += time_cost
            print(f"kwargs: {kwargs}")
            return ret
        return wrapper
    return decorator


def idle_collate(batch):
    return batch

class GrpoAlgorithm(BaseAlgorithm):
    """GrpoAlgorithm"""

    def __init__(self, cfg: GrpoConfig) -> None:
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
        if self.cfg.runtime_args.train_backend == "fsdp":
            policy_trainer = PolicyTrainer("policy_trainer")
            ref_policy = PolicyTrainer("ref_policy")
        elif self.cfg.runtime_args.train_backend == "megatron":
            policy_trainer = MegatronPolicyTrainer("policy_trainer")
            ref_policy = MegatronPolicyTrainer("ref_policy")

        # setup for rollout
        if self.cfg.runtime_args.task_type == "chat":
            if self.cfg.runtime_args.rollout_backend == "vllm":
                policy = VLLMModule("policy")
            elif self.cfg.runtime_args.rollout_backend == "sglang":
                rollout_cls = SGLangModule if self.cfg.models.policy.is_sync_mode else AsyncSGLangModule
                policy = rollout_cls("policy")
        elif self.cfg.runtime_args.task_type == "agent":
            assert not self.cfg.models.policy.is_sync_mode and self.cfg.runtime_args.rollout_backend == "sglang", \
                "agent task only support async sglang engine"
            assert self.cfg.runtime_args.use_rollout_manager, "agent task must set use_rollout_manager=True"
            policy = AgentModule("policy")

        reward = RuleReward("reward")
        #rollout_manager = RolloutManager("rollout_manager") if self.cfg.runtime_args.use_rollout_manager else None
        #partial_rollout_manager =  PartialRolloutManager("partial_rollout_manager") if self.cfg.runtime_args.use_partial_rollout else None

        return {"policy": policy, "reward": reward, "ref_policy": ref_policy, "policy_trainer": policy_trainer}
    
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
        # Ceate resources
        model_list = [v for k,v in self.models.items()]
        resource_manager = ResourceManager(model_list)
        param_sync_pairs = {'policy_trainer': 'policy'}
        # Create ray actors
        self.model_manager = ModelManager(model_list, resource_manager, self.cfg)
        self.model_manager.create_dist_models()
        self.remote_models = self.model_manager.dist_models
        self.named_models = {model.name: model for model in self.remote_models}
        for src, dst in param_sync_pairs.items():
            self.model_manager.set_parameter_sync(self.models[src], self.models[dst])
        all_actors = ray.util.list_named_actors(all_namespaces=True)
        print(all_actors)
        for model in self.remote_models:
            model.init()
        
        # init engine actor for rollout actors
        refs = []
        for replica in self.named_models['policy'].replicas:
            if self.cfg.runtime_args.rollout_backend == 'vllm':
                refs.append(replica.setup_engine(replica.all_actors))
            else:
                refs.append(replica.setup_engine())
        future.wait(refs, return_output=True)
        self.named_models['policy'].offload()

        # init train actor and other rollout actors
        for model in self.remote_models:
            future.wait(model.model_setup(), return_output=True)
        for model in self.remote_models:
            model.group_dist_actors_by_dp_rank()
        self.num_dp_ranks = {k: sum([len(replica.dp_rank_to_actors) for replica in v.replicas]) for k, v in self.named_models.items()}

        # Build parameter sync group and do first sync
        self.model_manager.build_parameter_group()
        self.model_manager.sync_parameters()

    def create_data_iterator(self, data_list, num_batch):
        slice_index = even_slice(len(data_list), num_batch)
        batches = []
        for start, end in zip(slice_index[:-1], slice_index[1:]):
            batches.append(data_list[start:end])
        return iter(batches)

    @timeit()
    def run_model_one_round(self, data, model_name, func_name):
        # This function is used to run one step for all actors
        dp_size = self.num_dp_ranks[model_name]
        data_iter = self.create_data_iterator(data, dp_size)
        refs = []
        if model_name == 'policy':
            for replica in self.named_models[model_name].replicas:
                batches = next(data_iter)
                ref = replica.forward_step(*[batches])
                refs.extend(ref)
        else:
            # all actors in same dp rank have same input
            for replica in self.named_models[model_name].replicas:
                for _, actors in replica.dp_rank_to_actors.items():
                    batches = next(data_iter)
                    for actor in actors:
                        ref = replica.call_actor_remote_func(actor, func_name, batches)
                        # output length is num actor
                        refs.append(ref)
        result = ray.get(refs)
        if isinstance(result[0], list):
            result = list(itertools.chain.from_iterable(result))
        return result

    def eval(self):
        # Get eval data
        data = next(self.eval_dataiter)
        data = self.run_model_one_round(data, model_name='policy', func_name='forward_step')
        #self.named_models['policy'].offload()
        data = self.run_model_one_round(data, model_name='reward', func_name='forward_step')
        eval_metric_list = self.gather_metrics("reward")
        print(eval_metric_list)
        eval_rollout_metric_list = self.gather_metrics("policy")
        print(eval_rollout_metric_list)
        return True

    def train(self):
        data = next(self.train_dataiter)
        # Rollout
        self.named_models['policy'].onload()
        data = self.run_model_one_round(data, model_name='policy', func_name='forward_step')
        total_valid_tokens = sum(d['response_token_length'] + d['prompt_token_length'] for d in data)
        self.named_models['policy'].offload()
        self.gather_metrics("policy")

        # Reward/Adv
        data = self.run_model_one_round(data, model_name='reward', func_name='forward_step')
        self.gather_metrics("reward")
        data = self.adv_computer(data)

        # calculate total_minibsz needed. num_microbsz should be divided by total_minibsz
        dp_size = self.num_dp_ranks['policy_trainer']
        num_minibsz = self.cfg.runtime_args.sample_per_episode // self.cfg.runtime_args.train_global_batch_size
        total_minibsz = num_minibsz * dp_size
        # Split list of sample to list of microbatches
        packed_data = regroup_data_packing_simple(data, total_minibsz, self.cfg.models.policy_trainer.max_token_in_packing, 0)
        packed_data = rearrange_zigzag(packed_data, total_minibsz)
        # ref_logprobs
        self.named_models['ref_policy'].onload()
        data = self.run_model_one_round(packed_data, model_name='ref_policy', func_name='forward_step')
        self.named_models['ref_policy'].offload()

        # old_logprobs
        self.named_models['policy_trainer'].onload()
        data = self.run_model_one_round(data, model_name='policy_trainer', func_name='forward_step')
        # import pdb
        # pdb.set_trace()

        # train
        minibsz_iter = self.create_data_iterator(data, num_minibsz)
        for minibsz_idx in range(num_minibsz):
            mini_batch_data = next(minibsz_iter)
            data = self.run_model_one_round(mini_batch_data, model_name='policy_trainer', func_name='train_step')
        self.named_models['policy_trainer'].offload()
        self.gather_metrics("policy_trainer")
        return total_valid_tokens

    def gather_metrics(self, model_name):
        metric_list = []
        for replica in self.named_models[model_name].replicas:
            for _, actors in replica.dp_rank_to_actors.items():
                ref = replica.call_actor_remote_func(actors[-1], "get_and_clear_metrics")
                # output length is num actor
                metric_list.append(ref)
        metric_list = ray.get(metric_list)
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
            start = time.time()
            total_tokens = self.train()
            self.model_manager.sync_parameters()
            end = time.time()
            self._timing_summary['episode_e2e'] = end - start
            self._timing_summary['tps'] = total_tokens / (end - start)
            print(self._metrics_gather)
            print(self._timing_summary)
            for k, v in self._metrics_gather.items():
                self.metric_manager.log(k, episode_id, v)
            self.metric_manager.log('train_time_summary', episode_id, self._timing_summary)
            self._timing_summary.clear()
            self._metrics_gather = {}
            if episode_id % self.cfg.runtime_args.eval_episode_interval == 0:
                start = time.time()
                self.eval()
                end = time.time()
                self._timing_summary['eval_e2e'] += end - start
                for k, v in self._metrics_gather.items():
                    self.metric_manager.log(f"eval_{k}", episode_id, v)
                self.metric_manager.log('eval_time_summary', episode_id, self._timing_summary)
                self._timing_summary.clear()
                self._metrics_gather = {}

        # put data in engine._all_datasets
    def validate(self):
        self.cfg.validate()
