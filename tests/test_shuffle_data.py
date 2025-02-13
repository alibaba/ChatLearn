import os
from chatlearn.data.data import RLHFDataLoader

def test_case1():
    train_prompts = ['a', 'b', 'c', 'd', 'e', 'f']
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

    data_queue = []
    shuffled_data = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e']
    dataloader = RLHFDataLoader(train_prompts, batch_size=2, num_inference_per_prompt=2, shuffled_list=shuffled_data, offset=1)
    data_iter = iter(dataloader)
    for i in range(8):
        data_queue.append(next(data_iter))
        pass
    assert(data_queue[0] == ['b', 'b', 'c', 'c'] and data_queue[1] == ['d', 'd', 'e', 'e'])
    new_queue = []
    for data in data_queue[2:]:
        new_queue.extend(data)
    # print(new_queue)
    assert(not(new_queue[:12] == shuffled_data and new_queue[12:] == shuffled_data))

def test_case2():
    train_prompts = ['a', 'b', 'c', 'd', 'e']
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
    sorted_queue = []
    for q in data_queue:
        tmp = []
        for b in q:
            tmp.extend(b)
        sorted_queue.append(sorted(tmp))
    assert(sorted_queue[0] == sorted_queue[1] and sorted_queue[0] == sorted_queue[2])
    # print(sorted_queue)

    data_queue = []
    shuffled_data = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e']
    dataloader = RLHFDataLoader(train_prompts, batch_size=2, num_inference_per_prompt=2, shuffled_list=shuffled_data, offset=1)
    data_iter = iter(dataloader)
    for i in range(8):
        data_queue.append(next(data_iter))
        pass
    assert(data_queue[0] == ['b', 'b', 'c', 'c'] and data_queue[1] == ['d', 'd', 'e', 'e'])
    new_queue = []
    for data in data_queue[2:]:
        new_queue.extend(data)
    # print(new_queue)
    assert(not(new_queue[:12] == shuffled_data and new_queue[12:] == shuffled_data))


if __name__ == "__main__":
    test_case1()
    test_case2()