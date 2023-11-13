# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
"""test vllm policy generation"""

from models.vllm_policy_inference import VLLMPolicyInference
from models.utils import write_jsonl
from tqdm import tqdm
from train_rlhf import get_prompts

import chatlearn
from chatlearn import EvalEngine

chatlearn.init()

policy = VLLMPolicyInference("policy")
policy.register_eval_func("forward_step")
engine = EvalEngine(policy)

args = chatlearn.get_args()
k = {"math_coef": 0}
train_prompts = get_prompts(args.rlhf_args.get("eval_data_path"), num_limit=512, )

policy_checkpoint = policy.model_args["load"]
load_iteration = policy.model_args.get("load_iteration", 0)
exp_name = policy.model_args["exp_name"]
eval_dir = args.rlhf_args._args_dict["eval_output_dir"]

engine.set_dataset(train_prompts)
results = engine.eval()
output = []

for res in tqdm(results, total=len(results)):
    print(res['str_outputs'])
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
