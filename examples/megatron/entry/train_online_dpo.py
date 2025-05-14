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
"""entry file for training online dpo"""

import random
import os

from examples.megatron.models import PolicyReference, PolicyTrainer, RewardInference
from examples.megatron.models.utils import get_prompts
from examples.megatron.models.eval_post_process import EvaluatorPostProcess

import chatlearn
from chatlearn import OnlineDPOEngine

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
    reference_model = PolicyReference("reference")
    policy_trainer = PolicyTrainer("ppo_policy")

    policy_model = PolicyModel("policy")
    reward_model = RewardInference("reward")

    engine = OnlineDPOEngine(policy_model, reference_model, reward_model, policy_trainer)

    all_prompts = get_prompts(args.runtime_args.data_path, num_limit=args.runtime_args._args_dict['training_data_num_limit'])
    random.seed(reference_model.model_args["seed"])
    split_ratio = 0.9 if args.runtime_args.eval_episode_interval > 0 else 1
    num_train = int(len(all_prompts) * split_ratio)
    random.shuffle(all_prompts)
    train_prompts = all_prompts[:num_train]

    if args.runtime_args.eval_episode_interval > 0:
        val_prompts = all_prompts[num_train:]
        eval_num_limit = args.runtime_args.get('eval_data_num_limit')
        if eval_num_limit:
            eval_num_limit = min(eval_num_limit, len(val_prompts))
            val_prompts = val_prompts[:eval_num_limit]

        def eval_flow(batch):
            r0 = policy_model.eval_forward(batch)
            r1 = reward_model.eval_forward(r0)
            return r1

        evaluator = EvaluatorPostProcess(eval_flow) \
            .set_dataset(val_prompts)
        engine.set_evaluator(evaluator)
    engine.set_dataset(train_prompts)
    engine.learn()
