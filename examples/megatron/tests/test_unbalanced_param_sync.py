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

from examples.megatron.models import PolicyTrainer
from examples.megatron.models.train_helper import get_prompts

import chatlearn
from chatlearn.models.base_module import BaseModule
from chatlearn.runtime.engine import Engine
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.trainer import Trainer

# pylint: disable=invalid-envvar-default,bad-exception-cause,ungrouped-imports
if os.getenv("ENABLE_VLLM", False):
    try:
        from examples.megatron.models import VLLMPolicyInference as PolicyModel
    except Exception as e:
        raise RuntimeError("Cannot import vllm, please set vllm python path or install vllm first.") from e
else:
    from examples.megatron.models import PolicyInference as PolicyModel


class CustomEngine(Engine):
    """Custom engine for param sync from ppo_policy to policy."""
    def __init__(self,
                 policy: BaseModule,
                 policy_trainer: BaseModule):
        def env_compute_flow(batch):
            policy_out = policy.forward_step(batch)
            return policy_out

        def trainer_compute_flow(batch):
            policy_trainer.train_step(batch)

        env = Environment(env_compute_flow)
        trainer = Trainer(trainer_compute_flow)
        super().__init__(env, trainer, name='ParamSync')
        self.set_parameter_sync(policy_trainer, policy)


if __name__ == "__main__":
    chatlearn.init()
    args = chatlearn.get_args()
    ppo_policy = PolicyTrainer("ppo_policy")
    policy_model = PolicyModel("policy")

    engine = CustomEngine(policy_model, ppo_policy)
    train_prompts = get_prompts(args.runtime_args.data_path, num_limit=args.runtime_args._args_dict['training_data_num_limit'])
    engine.set_dataset(train_prompts)
    engine.learn()
