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

import os
import random
from collections import defaultdict
import numpy

from examples.megatron.models import PolicyReference, PolicyTrainer, RewardInference
from examples.megatron.models.reward_math import MathReward
from examples.megatron.models.train_helper import eval_post_process, get_prompts

import chatlearn
from chatlearn import GRPOMathEngine


# pylint: disable=invalid-envvar-default,bad-exception-cause,ungrouped-imports
if os.getenv("ENABLE_VLLM", False):
    try:
        from examples.megatron.models import VLLMPolicyInference as PolicyModel
    except Exception as e:
        raise RuntimeError("Cannot import vllm, please set vllm python path or install vllm first.") from e
else:
    from examples.megatron.models import PolicyInference as PolicyModel

# GRPO advantage计算relay_sample_fn
def grpo_rw_relay_sample_fn(episode_relay_buffers):
    buffers = episode_relay_buffers[-1].buffer
    queryids2samples = defaultdict(list)
    for s in buffers:
        queryids2samples[str(s["no_padded_query_ids"].cpu().tolist())].append(s)

    math_reward_strategy = math_reward_model.model_args['math_reward_strategy']

    res_buffers = []
    for _, l in queryids2samples.items():
        if math_reward_strategy == 'sparse_only':
            assert "math_rewards" in l[0], l[0].keys()
            rewards = [each["math_rewards"] for each in l]
        elif math_reward_strategy == 'dense_only':
            assert "rm_rewards" in l[0], l[0].keys()
            rewards = [each["action_rewards"] for each in l]
        elif math_reward_strategy == 'merge':
            assert "rm_rewards" in l[0] and "math_rewards" in l[0], l[0].keys()
            rewards = [each["math_rewards"] + each["rm_rewards"] for each in l]
        mean = numpy.mean(rewards)
        std = numpy.std(rewards)
        for i, li in enumerate(l):
            li['final_rewards'] = rewards[i]
            li['advantages'] = ((rewards[i] - mean) / (std + 1e-5))
        res_buffers.extend(l)
    assert len(buffers) == args.runtime_args.sample_per_episode
    return res_buffers

if __name__ == "__main__":
    chatlearn.init()
    args = chatlearn.get_args()
    reference_model = PolicyReference("reference")
    policy_trainer = PolicyTrainer("ppo_policy")

    policy_model = PolicyModel("policy")
    reward_model = RewardInference("reward")
    math_reward_model = MathReward("math_reward")

    engine = GRPOMathEngine(policy_model, reference_model, reward_model, math_reward_model, policy_trainer)

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
        engine.evaluator.set_dataset(val_prompts) \
            .set_post_process_func(eval_post_process)
    engine.set_dataset(train_prompts)
    engine.set_relay_sample_fn(grpo_rw_relay_sample_fn)
    engine.learn()
