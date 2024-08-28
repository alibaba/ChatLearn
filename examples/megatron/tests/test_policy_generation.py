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
"""test policy generation"""

import os
from tqdm import tqdm

from examples.megatron.models.utils import write_jsonl
from examples.megatron.models.train_helper import get_prompts
import chatlearn
from chatlearn import EvalEngine

# pylint: disable=invalid-envvar-default,bad-exception-cause,ungrouped-imports
if os.getenv("ENABLE_VLLM", False):
    try:
        from examples.megatron.models.vllm_policy_inference import VLLMPolicyInference as PolicyModel
    except Exception as e:
        raise RuntimeError("Cannot import vllm, please set vllm python path or install vllm first.") from e
else:
    from examples.megatron.models.old_policy_inference import PolicyInference as PolicyModel



chatlearn.init()


model_name = "policy"
policy = PolicyModel(model_name)
def eval_flow(batch):
    r0 = policy.eval_forward(batch)
    return r0

engine = EvalEngine(eval_flow)

args = chatlearn.get_args()
k = {"math_coef": 0}
train_prompts = get_prompts(args.runtime_args.get("eval_data_path"), num_limit=15)

policy_checkpoint = policy.model_args["load"]
load_iteration = policy.model_args.get("load_iteration", 0)
exp_name = args.runtime_args.exp_name
eval_dir = os.path.join(args.runtime_args.output_dir, "eval")

engine.set_dataset(train_prompts)
results = engine.eval()[model_name]
output = []

for res in tqdm(results, total=len(results)):
    print(res["str_outputs"])
    str_prompts = res["str_prompts"]
    str_outputs = res["str_outputs"]
    for str_prompt, str_output in zip(str_prompts, str_outputs):
        j = {"query": str_prompt, "responses": [str_output]}
        output.append(j)

policy_inference_fp = f"{eval_dir}/{load_iteration}/{exp_name}/inference_json.json"
print(policy_inference_fp)
print(f"inference finished: got jsons number: {len(output)}")
write_jsonl(output, policy_inference_fp)

engine.logging_summary()
