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
"""test reward forward"""

import json
import time
from collections import defaultdict

from tqdm import tqdm

from examples.megatron.models.reward_inference import RewardInference
import chatlearn
from chatlearn import Engine

chatlearn.init()

policy = RewardInference("reward")
engine = Engine(policy)
engine.setup()
model = engine.models[0].replicas[0]


def read_jsonl(file_path):
    with open(file_path, encoding="utf-8") as f1:
        return [json.loads(line) for line in f1]


def get_labled_list_strs(fp):
    jsons = read_jsonl(fp)
    result = []
    for item in jsons:
        samples = defaultdict(list)
        for i, response in enumerate(item['response']):
            samples['input'].append([item['query'], response])
            samples['score'].append(item['score'][i])
        result.append(samples)
    return result


args = chatlearn.get_args()
batches = get_labled_list_strs(args.runtime_args.get("eval_data_path"))

start_time = time.time()

right = 0
acc = 0
for batch in tqdm(batches):
    res = model.forward_step_pipeline(list_strs=batch['input'])  # [b,]
    res = chatlearn.get(res)
    # use last pipeline stage results
    res = res[-1]
    if res[0].item() > res[1].item():
        new_scores = [1, 0]
    else:
        new_scores = [0, 1]
    if new_scores == batch['score']:
        right += 1
    acc += 1
    print(f"current acc {right / acc}")

# expected 0.687046 tp=8
# final acc 0.6860670194003528 tp=4
# final acc 0.6870468351949833, tp=8, time takes 240.93485498428345s
print(f"final acc {right / acc}, time takes {time.time() - start_time}s")
