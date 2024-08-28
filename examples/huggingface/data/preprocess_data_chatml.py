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
"""
{
    "type": "chatml",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Tell me something about large language models."
        },
        {
            "role": "assistant",
            "content": "Large language models are a type of language model that is trained on a large corpus of text data. They are capable of generating human-like text and are used in a variety of natural language processing tasks..."
        }
    ],
    "source": "unknown"
}
"""

import json
import os
import sys

from datasets import load_dataset
from tqdm import tqdm
import re

task_type = 'alignment'
prefix = os.path.join(sys.argv[1], task_type)

def parse_dialogue(dialogue):
    items = [{"role": "system", "content": "You are a helpful assistant."}]
    # Define the patterns for Human and Assistant
    pattern = r'\n\n(Human|Assistant): (.*?)(?=\n\n(Human|Assistant):|\Z)'

    # Find all matches
    matches = re.findall(pattern, dialogue, re.DOTALL)

    for role, statement, _ in matches:
        if role == "Human":
            role = "user"
        elif role == "Assistant":
            role = "assistant"
        else:
            raise RuntimeError(f'unknown role {role}')
        items.append({"role": role, "content": statement.strip()})

    return items

while True:
    try:
        alignment_data = load_dataset('Dahoas/full-hh-rlhf')
        break
    except Exception as e:
        print(e)
        continue


if not os.path.exists(prefix):
    os.makedirs(prefix)

for split in ['train', 'test']:
    if split not in alignment_data:
        continue
    data = alignment_data[split]
    chatml_item = {}
    with open(f'{prefix}/{split}.jsonl', 'w') as f:
        for item in data:
            chatml_item["type"] = "chatml"
            input_text = item['prompt']
            # Parse and classify the messages
            chatml_item["messages"] = parse_dialogue(input_text)
            if task_type == 'sft':
                chatml_item["messages"].append({"role": "assistant", "content": item['response'].strip()})
            else:
                chatml_item["chosen"] = [{"role": "assistant", "content": item['chosen']}]
                chatml_item["rejected"] = [{"role": "assistant", "content": item['rejected']}]
            f.write(json.dumps(chatml_item) + '\n')
