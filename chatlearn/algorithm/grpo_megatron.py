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
import json
from typing import List
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from chatlearn.configs.common import (
    BaseConfig,
    RuntimeEnvConfig,
    PolicyConfig,
    RuntimeConfig,
    BaseModelConfig
)

from configs.megatron_config import MegatronRefPolicyConfig, MegatronPolicyTrainerConfig
from chatlearn.algorithm.base_algo import BaseAlgorithm
# from examples.mcore.entry.train_grpo import *
import chatlearn
from chatlearn import Evaluator
from chatlearn.utils.utils import listdict_to_dictlist
from chatlearn import Engine, Environment, Trainer
from chatlearn.algorithm.grpo_utils.vllm_policy_inference import VLLMPolicyInference
from chatlearn.models.reward.rule_reward import RuleReward
from chatlearn.algorithm.grpo_utils.megatron_policy_trainer import MegatronPolicyTrainer as PolicyTrainer


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

def read_data_path_list(data_path_list: List[str], mode: str = "jsonl"):

    data = []
    for data_path in data_path_list:
        if mode == "json":
            with open(data_path, 'r') as f:
                data.extend(json.load(f))
        elif mode == "jsonl":
            with open(data_path, 'r') as f:
                data.extend([json.loads(line) for line in f])
    return data

def compute_grpo_adv(episode_replay_buffers):
    buffers = episode_replay_buffers[-1].buffer
    queryids2samples = defaultdict(list)
    for s in buffers:
        queryids2samples[hash(','.join(map(str, s["prompt_token_ids"])))].append(s)
    
    res_buffers = []
    for _, l in queryids2samples.items():
        rewards = [each["rule_rewards"] for each in l]
        rewards = torch.cat(rewards, dim=0)

        mean = torch.mean(rewards)
        std = torch.std(rewards)
        for i, li in enumerate(l):
            li['advantages'] = ((rewards[i] - mean) / (std + 1e-5))
        res_buffers.extend(l)
    return res_buffers

class GRPOEvaluator(Evaluator):

    def post_process(self, results, eval_info):
        # results Dict[List]
        results = results["reward"]
        results = listdict_to_dictlist(results)
        # convert list[tensor(n,1)] to list[float]
        rule_rewards = results.get("rule_rewards", [])
        rule_rewards_flatten = torch.cat(rule_rewards).squeeze().tolist()

        reward_score = sum(rule_rewards_flatten) / len(rule_rewards_flatten)
        eval_reward_stats = {"eval_reward_score": reward_score}


        self._metric_list.append(eval_reward_stats)
        
        return results


class GRPOEngine(Engine):
    """GRPO Engine."""
    def __init__(self,
                 policy: VLLMPolicyInference,
                 reward: RuleReward,
                 ref_policy: PolicyTrainer,
                 policy_trainer: PolicyTrainer):
        def env_compute_flow(batch):
            policy_out = policy.forward_step(batch)
            old_logprobs_out = policy_trainer.forward_step(policy_out)
            ref_logprobs_out = ref_policy.forward_step(old_logprobs_out)
            reward_out = reward.forward_step(ref_logprobs_out)
            return ref_logprobs_out, reward_out

        def trainer_compute_flow(batch):
            policy_trainer.train_step(batch)

        def evaluator_flow(batch):
            policy_out = policy.forward_step(batch)
            reward_out = reward.eval_forward(policy_out)
            return reward_out

        env = Environment(env_compute_flow)
        trainer = Trainer(trainer_compute_flow)
        evaluator = GRPOEvaluator(evaluator_flow)
        super().__init__(environment=env, trainer=trainer, evaluator=evaluator, name='grpo')
        self.set_parameter_sync(policy_trainer, policy)

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
        