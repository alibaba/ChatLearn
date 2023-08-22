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
"""prepare data for rlhf"""

import json
import os
import sys

from datasets import load_dataset
from tqdm import tqdm

while True:
    try:
        rlhf_data = load_dataset('Dahoas/full-hh-rlhf')
        break
    except Exception as e:
        print(e)
        continue

prefix = os.path.join(sys.argv[1], 'rlhf')
if not os.path.exists(prefix):
    os.makedirs(prefix)
with open(os.path.join(prefix, 'train.jsonl'), 'w', encoding="utf-8") as f:
    for item in tqdm(rlhf_data['train']):
        item.pop('chosen')
        f.write(json.dumps(item) + '\n')
with open(os.path.join(prefix, 'dev.jsonl'), 'w', encoding="utf-8") as f:
    for item in tqdm(rlhf_data['test']):
        item.pop('chosen')
        f.write(json.dumps(item) + '\n')