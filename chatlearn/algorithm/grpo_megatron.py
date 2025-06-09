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

from dataclasses import dataclass, field
from chatlearn.configs.common import (
    BaseConfig,
    RuntimeEnvConfig,
    PolicyConfig,
    RuntimeConfig,
    # RefPolicyConfig,
    # PolicyTrainerConfig,
    BaseModelConfig
)

from configs.megatron_config import MegatronRefPolicyConfig, MegatronPolicyTrainerConfig
from chatlearn.algorithm.base_algo import BaseAlgorithm
from examples.mcore.entry.train_grpo import *



@dataclass
class GrpoModelConfig(BaseConfig):
    policy: PolicyConfig = field(
        default_factory=PolicyConfig,
        metadata={"help": "Policy config."}
    )
    reward: BaseModelConfig = field(
        default_factory=BaseModelConfig,
        metadata={"help": "Reward config."}
    )
    ref_policy: MegatronRefPolicyConfig = field(
        default_factory=MegatronRefPolicyConfig,
        metadata={"help": "Reference policy config."}
    )
    policy_trainer: MegatronPolicyTrainerConfig = field(
        default_factory=MegatronPolicyTrainerConfig,
        metadata={"help": "Policy trainer config."}
    )

@dataclass
class GrpoConfigMegatron(BaseConfig):
    """GrpoConfig"""

    env_args: RuntimeEnvConfig = field(
        default_factory=RuntimeEnvConfig,
        metadata={"help": "Runtime environment config."}
    )
    runtime_args: RuntimeConfig = field(
        default_factory=RuntimeConfig,
        metadata={"help": "Runtime config."}
    )
    models: GrpoModelConfig = field(
        default_factory=GrpoModelConfig,
        metadata={"help": "Grpo model config."}
    )



class GrpoAlgorithmMegatron(BaseAlgorithm):
    """GrpoAlgorithm"""

    def __init__(self, cfg: GrpoConfigMegatron) -> None:
        self.cfg = cfg


    def run(self) -> None:
        # print(self.cfg)
        # exit()
        chatlearn.init(self.cfg)
        args = chatlearn.get_args()
        policy_trainer = PolicyTrainer("policy_trainer")
        ref_policy = PolicyTrainer("ref_policy")
        policy = VLLMPolicyInference("policy")
        reward = RuleReward("reward")
        engine = GRPOEngine(policy, reward, ref_policy, policy_trainer)

        # get train and evaluation data
        train_data_path_list = [item.strip() for item in args.runtime_args.data_path.split(",")]
        train_data = read_data_path_list(train_data_path_list)

        eval_data_path_list = [item.strip() for item in args.runtime_args.eval_data_path.split(',')]
        eval_data = read_data_path_list(eval_data_path_list)

        # put data in engine._all_datasets
        engine.set_dataset(train_data)
        engine.evaluator.set_dataset(eval_data)
        engine.set_replay_sample_manager(compute_grpo_adv)
        engine.learn()
        