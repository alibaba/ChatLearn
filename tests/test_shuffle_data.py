import os
from chatlearn.data.data import RLHFDataLoader

if __name__ == "__main__":
    # chatlearn.init()
    # args = chatlearn.get_args()
    # ppo_policy = PolicyTrainer("ppo_policy")
    # policy_model = PolicyModel("policy")

    # engine = CustomEngine(policy_model, ppo_policy)
    train_prompts = ['a', 'b', 'c', 'd', 'e', 'f']
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
    assert(not(data_queue[0] == data_queue[1] and data_queue[0] == data_queue[2]))
    sorted_queue = []
    for q in data_queue:
        tmp = []
        for b in q:
            tmp.extend(b)
        sorted_queue.append(sorted(tmp))
    assert(sorted_queue[0] == sorted_queue[1] and sorted_queue[0] == sorted_queue[2])
    # print(sorted_queue)