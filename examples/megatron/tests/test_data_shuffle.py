import os
from chatlearn.data.data import RLHFDataLoader
import json
from pathlib import Path

def read_jsonl(file_path):
    print(f"read_jsonl from : {file_path}")

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


    with open(file_path, encoding="utf-8") as f1:
        res = [json.loads(line) for line in f1]

    return res

def get_prompts_chatml_formatted(fp, num_limit=-1):
    prompts_jsons = read_jsonl(fp)
    for j in prompts_jsons:
        if "system" not in j:
            j["system"] = ""
    assert num_limit <= len(prompts_jsons), f"{num_limit} <= {len(prompts_jsons)}"

    if num_limit != -1:
        prompts_jsons = prompts_jsons[:num_limit]

    return prompts_jsons

if __name__ == "__main__":
    # chatlearn.init()
    # args = chatlearn.get_args()
    # ppo_policy = PolicyTrainer("ppo_policy")
    # policy_model = PolicyModel("policy")

    # engine = CustomEngine(policy_model, ppo_policy)
    train_prompts = [{'a'}, {'b'}, {'c'}, {'d'}, {'e'}, {'f'}]
    # train_prompts = get_prompts_chatml_formatted("/cpfs01/user/zhengchujie.zcj/Math-Data-Filter/collected_afterNY_rl_queries_selected.jsonl", num_limit=6)
    dataloader = RLHFDataLoader(train_prompts, batch_size=2, num_inference_per_prompt=2)
    data_queue, data_queue_tmp = [], []
    data_iter = iter(dataloader)
    for i in range(9):
        data = next(data_iter)
        data_queue_tmp.append(data)
        if (i + 1) % 3 == 0:
            data_queue.append(data_queue_tmp)
            data_queue_tmp = []
    
    # print(data_queue)
    assert(not(data_queue[0] == data_queue[1] and data_queue[0] == data_queue[2]))

