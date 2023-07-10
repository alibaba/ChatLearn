import json
from collections import defaultdict

from reward_inference_module_hongyi import RewardModelMegatronInference
from tqdm import tqdm

import rlhf
from rlhf import Engine

rlhf.init()

policy = RewardModelMegatronInference("reward")
engine = Engine(policy)
engine.setup()
model = engine.models[0].replicas[0]


def read_jsonl(file_path):
    with open(file_path, encoding="utf-8") as f1:
        res = [json.loads(line) for line in f1]
        return res


def get_labled_list_strs(fp):
    jsons = read_jsonl(fp)
    res = []
    for item in jsons:
        batch = defaultdict(list)
        for i, response in enumerate(item['response']):
            batch['input'].append([item['query'], response])
            batch['score'].append(item['score'][i])
        res.append(batch)
    return res


fp = "/cpfs01/shared/Group-m6/yuanhongyi.yhy/anthropic-hh/rm/dev.jsonl"
batches = get_labled_list_strs(fp)

right = 0
acc = 0
for batch in tqdm(batches):
    res = model.forward_step_pipeline(batch['input'])  # [b,]
    res = rlhf.get(res)
    res = res[0]
    if res[0].item() > res[1].item():
        new_scores = [1, 0]
    else:
        new_scores = [0, 1]
    print(res[0], "vs", batch['score'])
    if new_scores == batch['score']:
        right += 1
    acc += 1
    print(f"")

# expected 0.6147
print(f"acc {right / len(batches)}")
