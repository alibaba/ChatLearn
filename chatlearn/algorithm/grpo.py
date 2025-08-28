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
    PolicyConfig,
    RuntimeConfig,
    RuntimeEnvConfig
)
from chatlearn.configs.fsdp_config import FSDPPolicyTrainerConfig, FSDPRefPolicyConfig

from chatlearn.algorithm.grpo_utils.advantage_compute import compute_grpo_adv
from chatlearn.algorithm.grpo_utils.policy_trainer import PolicyTrainer
from chatlearn.algorithm.grpo_utils.vllm_policy_inference import \
    VLLMPolicyInference
from chatlearn.algorithm.grpo_utils.sglang_policy_inference import SGLangPolicyInference, AsyncSGLangPolicyInference
from chatlearn.data.data import read_data_path_list
from chatlearn.models.reward.rule_reward import RuleReward
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
            assert sample_per_episode % (conf.num_replica * conf.replica_dp_size) == 0, (
                f"runtime_args.sample_per_episode must be divisible by models.{name}.num_replica times models.{name}.replica_dp_size, "
                f"but {name} got num_replica: {conf.num_replica}; replica_dp_size: {conf.replica_dp_size}"
            )
            sample_per_dp_rank = sample_per_episode // (conf.num_replica * conf.replica_dp_size)
            assert sample_per_dp_rank % conf.generation_batch_size == 0, (
                f"sample_per_dp_rank must be divisible by models.{name}.generation_batch_size, "
                f"but {name} got sample_per_dp_rank: {sample_per_dp_rank}; generation_batch_size: {conf.generation_batch_size}"
            )

            # TODO: add checks after dataflow refactorization
            # NOTE: each rank will get `generation_batch_size` samples, and then:
            # mcore: use micro_batch_size=train_micro_batch_size
            # fsdp: use micro_batch_size=generation_batch_size
            # we skip these checks currently

            train_global_batch_size_per_dp_rank = self.runtime_args.train_global_batch_size // (conf.num_replica * conf.replica_dp_size)
            assert train_global_batch_size_per_dp_rank % self.runtime_args.train_micro_batch_size == 0, (
                f"sample_per_dp_rank must be divisible by runtime_args.train_micro_batch_size, "
                f"but {name} got sample_per_dp_rank: {sample_per_dp_rank}; train_micro_batch_size: {self.runtime_args.train_micro_batch_size}"
            )



class GRPOEvaluator(Evaluator):
    """GRPOEvaluator"""

    def post_process(self, results, eval_info):
        eval_reward_stats = defaultdict(list)
        for batch in results["reward"]:
            for result in batch:
                key = result["eval_source"]
                eval_reward_stats["eval_"+key].append(result["rule_reward"])
        for key in eval_reward_stats:
            eval_reward_stats[key] = sum(eval_reward_stats[key]) / len(eval_reward_stats[key])
        self._metric_list.append(eval_reward_stats)

        return results


class GRPOEngine(Engine):
    """GRPO Engine."""

    def __init__(
        self,
        policy: VLLMPolicyInference,
        reward: RuleReward,
        ref_policy: PolicyTrainer,
        policy_trainer: PolicyTrainer,
    ):
        def env_compute_flow(batch):
            policy_out = policy.forward_step(batch)
            old_logprobs_out = policy_trainer.forward_step(policy_out)
            ref_logprobs_out = ref_policy.forward_step(old_logprobs_out)
            reward_out = reward.forward_step(ref_logprobs_out)
            return reward_out

        def trainer_compute_flow(batch):
            policy_trainer.train_step(batch)

        def evaluator_flow(batch):
            policy_out = policy.eval_forward(batch)
            reward_out = reward.eval_forward(policy_out)
            return reward_out

        env = Environment(env_compute_flow)
        trainer = Trainer(trainer_compute_flow)
        evaluator = GRPOEvaluator(evaluator_flow)
        super().__init__(
            environment=env, trainer=trainer, evaluator=evaluator, name="grpo"
        )
        self.set_parameter_sync(policy_trainer, policy)


class GrpoAlgorithm(BaseAlgorithm):
    """GrpoAlgorithm"""

    def __init__(self, cfg: GrpoConfig) -> None:
        if cfg.runtime_args.train_backend == "megatron":
            cfg = update_cfg(cfg)

        self.cfg = cfg

    def run(self) -> None:
        chatlearn.init(self.cfg)

        if self.cfg.runtime_args.train_backend == "fsdp":
            policy_trainer = PolicyTrainer("policy_trainer")
            ref_policy = PolicyTrainer("ref_policy")
        elif self.cfg.runtime_args.train_backend == "megatron":
            policy_trainer = MegatronPolicyTrainer("policy_trainer")
            ref_policy = MegatronPolicyTrainer("ref_policy")
        if self.cfg.runtime_args.rollout_backend == "vllm":
            policy = VLLMPolicyInference("policy")
        elif self.cfg.runtime_args.rollout_backend == "sglang":
            RolloutModule_cls = SGLangPolicyInference if self.cfg.models.policy.is_sync_mode else AsyncSGLangPolicyInference
            policy = RolloutModule_cls("policy")
        reward = RuleReward("reward")
        engine = GRPOEngine(policy, reward, ref_policy, policy_trainer)

        # get train and evaluation data
        train_data_path_list = [
            item.strip() for item in self.cfg.runtime_args.data_path.split(",")
        ]
        train_data = read_data_path_list(train_data_path_list)

        eval_data_path_list = [
            item.strip() for item in self.cfg.runtime_args.eval_data_path.split(",")
        ]
        eval_data = read_data_path_list(eval_data_path_list)

        # put data in engine._all_datasets
        engine.set_dataset(train_data)
        engine.evaluator.set_dataset(eval_data)
        engine.set_replay_sample_manager(compute_grpo_adv)
        engine.learn()

    def validate(self):
        self.cfg.validate()
