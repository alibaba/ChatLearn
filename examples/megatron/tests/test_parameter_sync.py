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
"""entry file for training RLHF"""

import os

from examples.megatron.models import PolicyReference, PolicyTrainer, RewardInference, ValueInference, ValueTrainer

import chatlearn
from chatlearn import RLHFEngine

# pylint: disable=invalid-envvar-default,bad-exception-cause,ungrouped-imports
if os.getenv("ENABLE_VLLM", False):
    try:
        from examples.megatron.models import VLLMPolicyInference as PolicyModel
    except Exception as e:
        raise RuntimeError("Cannot import vllm, please set vllm python path or install vllm first.") from e
else:
    from examples.megatron.models import PolicyInference as PolicyModel


if __name__ == "__main__":
    chatlearn.init()
    args = chatlearn.get_args()
    args.runtime_args.debug = True
    reference_model = PolicyReference("reference")
    policy_trainer = PolicyTrainer("ppo_policy")

    policy_model = PolicyModel("policy")
    reward_model = RewardInference("reward")

    value_model = ValueInference("value")
    value_trainer = ValueTrainer("ppo_value")
    engine = RLHFEngine(policy_model, reference_model, reward_model, value_model, policy_trainer, value_trainer)
    train_prompts = ['test'] * 2048
    engine.set_dataset(train_prompts)
    engine.learn()
