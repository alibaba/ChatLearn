import json
import time
from collections import defaultdict

from models.reward_inference import RewardModelMegatronInference
from tqdm import tqdm

import chatlearn
from chatlearn import Engine

chatlearn.init()

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

args = chatlearn.get_args()
batches = get_labled_list_strs(args.rlhf_args.get("eval_data_path"))


start_time = time.time()

right = 0
acc = 0
for batch in tqdm(batches):
    res = model.forward_step_pipeline(batch['input'])  # [b,]
    res = chatlearn.get(res)
    res = res[0]
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
