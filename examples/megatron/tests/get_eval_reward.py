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
"""eval examples."""

import os

# pylint: disable=invalid-envvar-default,bad-exception-cause,ungrouped-imports,missing-module-docstring,wrong-import-position
if os.getenv("ENABLE_VLLM", False):
    try:
        from examples.megatron.models.vllm_policy_inference import VLLMPolicyInference as PolicyModel
    except Exception as e:
        raise RuntimeError("Cannot import vllm, please set vllm python path or install vllm first.") from e
else:
    from examples.megatron.models.old_policy_inference import PolicyInference as PolicyModel

from examples.megatron.models.reward_inference import RewardInference
from examples.megatron.models.train_helper import eval_post_process, get_prompts

import chatlearn
from chatlearn import EvalEngine

chatlearn.init()

args = chatlearn.get_args()

policy = PolicyModel("policy")
exp_name = args.runtime_args.exp_name
reward_inference = RewardInference("reward")

val_prompts = get_prompts(args.runtime_args.get("eval_data_path"), num_limit=args.runtime_args.get("eval_data_num_limit"), )

def eval_flow(batch):
    r0 = policy.eval_forward(batch)
    r1 = reward_inference.eval_forward(r0)
    return r1

engine = EvalEngine(eval_flow)
engine.set_dataset(val_prompts).set_post_process_func(eval_post_process)

load_iteration = args.models['policy'].args_dict['load_iteration']
if not load_iteration:
    load_iteration = 1
results = engine.eval(train_iteration=load_iteration)

engine.logging_summary()

# validate all prompts are processed
res_prompts = []

for data in results['reward']:
    for eval_jsonl in data['eval_jsonl']:
        prompt = eval_jsonl['query']
        res_prompts.append(prompt)
assert len(res_prompts) == len(val_prompts), f"{len(res_prompts)} vs {len(val_prompts)}"
