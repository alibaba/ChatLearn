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
try:
    from chatlearn.utils.megatron_utils import update_cfg
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
