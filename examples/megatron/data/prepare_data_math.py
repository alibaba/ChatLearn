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
"""prepare data for math"""
import os
import sys
import json
from datasets import load_dataset
from tqdm import tqdm
dataset = load_dataset('openai/gsm8k', 'main')

prefix = os.path.join(sys.argv[1], 'math')
if not os.path.exists(prefix):
    os.makedirs(prefix)

for tag in dataset:
    with open(os.path.join(prefix, f'{tag}.jsonl'), 'w', encoding="utf-8") as f:
        for item in tqdm(dataset[tag]):
            prompt = f"\n\nHuman: {item['question']}\n\nAssistant: "
            new_item = {"eval_func": "math_rule", "prompt": prompt, 'answer': item['answer']}
            f.write(json.dumps(new_item) + '\n')
