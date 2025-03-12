import os
from chatlearn.data.data import RLHFDataLoader
from chatlearn.data.sampler import MultiDatasetSampler
from itertools import cycle

def collate_fn(batch):
    return batch

def single_dataset():
    # evaluation
    dataset = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    sampler_eval = MultiDatasetSampler([len(dataset)], 9, consumed_samples=0, num_inference_per_prompt=1, shuffle=False, is_eval=True)
    dataloader = RLHFDataLoader([dataset], sampler_eval, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    for i in range(5):
        batches = next(data_iter)
        ground_truth = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        # print(batches)
        assert batches == ground_truth
        data_iter = cycle(iter(dataloader))

    # training
    dataset1 = [1, 2]
    sampler_train = MultiDatasetSampler([2], 6, [1], consumed_samples=0, num_inference_per_prompt=2, shuffle=False, is_eval=False)
    dataloader = RLHFDataLoader(datasets=[dataset1], sampler=sampler_train, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches = []
    for i in range(5):
        batches.append(next(data_iter))
    ground_truth = [[1,1,2,2,1,1], [2,2,1,1,2,2], [1,1,2,2,1,1], [2,2,1,1,2,2], [1,1,2,2,1,1]]
    assert batches == ground_truth

    # checkpoint
    dataset1 = [1, 2]
    sampler_train = MultiDatasetSampler([2], 6, [1], consumed_samples=6, num_inference_per_prompt=2, shuffle=False, is_eval=False)
    dataloader = RLHFDataLoader(datasets=[dataset1], sampler=sampler_train, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches = []
    for i in range(5):
        batches.append(next(data_iter))
    ground_truth = [[2,2,1,1,2,2], [1,1,2,2,1,1], [2,2,1,1,2,2], [1,1,2,2,1,1], [2,2,1,1,2,2]]
    assert batches == ground_truth

def multiple_dataset():
    
    # evaluation
    dataset1 = [1, 2, 3, 4, 5]
    dataset2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    sampler_eval = MultiDatasetSampler([5, 9], 14, consumed_samples=0, num_inference_per_prompt=1, shuffle=False, is_eval=True)
    dataloader = RLHFDataLoader([dataset1, dataset2], sampler_eval, collate_fn=collate_fn)
    ground_truth = dataset1 + dataset2
    data_iter = cycle(iter(dataloader))
    for i in range(5):
        batches = next(data_iter)
        data_iter = cycle(iter(dataloader)) # reset
        assert batches == ground_truth

    # training
    sampler_train = MultiDatasetSampler([5, 9], 10, [1, 3], consumed_samples=0, num_inference_per_prompt=2, shuffle=True, is_eval=False)
    dataloader = RLHFDataLoader(datasets=[dataset1, dataset2], sampler=sampler_train, collate_fn=collate_fn)
    data_iter = iter(dataloader)
    sequence = []
    for i in range(10):
        it = next(data_iter)
        sequence.extend(it)
    # num_inference_per_prompt
    assert len(sequence) == 10 * 10

    for i in range(0, len(sequence) // 2, 2):
        assert sequence[i] == sequence[i + 1]

    # data checkpoint
    sampler_train = MultiDatasetSampler([5, 9], 10, [1, 3], consumed_samples=20, num_inference_per_prompt=2, shuffle=True, is_eval=False)
    dataloader = RLHFDataLoader(datasets=[dataset1, dataset2], sampler=sampler_train, collate_fn=collate_fn)
    data_iter = iter(dataloader)
    new_sequence = []
    for i in range(10):
        it = next(data_iter)
        new_sequence.extend(it)
    assert sequence[20:] == new_sequence[:len(sequence) - 20]

def multi_replica():
    # evaluation
    dataset1 = [1, 2, 3, 4, 5]
    dataset2 = ['a', 'b']
    sampler_eval1 = MultiDatasetSampler([5, 2], 7, consumed_samples=0, num_inference_per_prompt=1, shuffle=False, is_eval=True, data_parallel_rank=0, data_parallel_size=2)
    dataloader1 = RLHFDataLoader([dataset1, dataset2], sampler_eval1, collate_fn=collate_fn)
    ground_truth1 = [1, 2, 3, 4]
    data_iter1 = cycle(iter(dataloader1))
    for i in range(5):
        batches = next(data_iter1)
        # print(batches)
        data_iter1 = cycle(iter(dataloader1)) # reset
        assert batches == ground_truth1

    sampler_eval2 = MultiDatasetSampler([5, 2], 7, consumed_samples=0, num_inference_per_prompt=1, shuffle=False, is_eval=True, data_parallel_rank=1, data_parallel_size=2)
    dataloader2 = RLHFDataLoader([dataset1, dataset2], sampler_eval2, collate_fn=collate_fn)
    ground_truth2 = [5, 'a', 'b']
    data_iter2 = cycle(iter(dataloader2))
    for i in range(5):
        batches = next(data_iter2)
        # print(batches)
        data_iter2 = cycle(iter(dataloader2)) # reset
        assert batches == ground_truth2

    # training
    dataset1 = [1, 2, 3, 4, 5]
    dataset2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    sampler_train = MultiDatasetSampler([5, 9], 20, [1, 3], consumed_samples=0, num_inference_per_prompt=2, shuffle=False, is_eval=False, data_parallel_rank=0, data_parallel_size=2)
    dataloader = RLHFDataLoader(datasets=[dataset1, dataset2], sampler=sampler_train, collate_fn=collate_fn)
    data_iter = iter(dataloader)
    sequence = []
    ground_truth = [1, 1, 'a', 'a', 'b', 'b', 'c', 'c', 2, 2, 'h', 'h', 'i', 'i', 4, 4, 'a', 'a', 'b', 'b', 1, 1, 'g', 'g', 'h', 'h', 'i', 'i', 2, 2]
    for i in range(3):
        sequence.extend(next(data_iter))
    assert sequence == ground_truth

    sampler_train = MultiDatasetSampler([5, 9], 20, [1, 3], consumed_samples=0, num_inference_per_prompt=2, shuffle=False, is_eval=False, data_parallel_rank=1, data_parallel_size=2)
    dataloader = RLHFDataLoader(datasets=[dataset1, dataset2], sampler=sampler_train, collate_fn=collate_fn)
    data_iter = iter(dataloader)
    sequence = []
    ground_truth = ['d', 'd', 'e', 'e', 'f', 'f', 3, 3, 'g', 'g', 'c', 'c', 5, 5, 'd', 'd', 'e', 'e', 'f', 'f', 'a', 'a', 'b', 'b', 'c', 'c', 3, 3, 'd', 'd']
    for i in range(3):
        sequence.extend(next(data_iter))
    assert sequence == ground_truth

def uid():
    # training
    dataset1 = [
        {"prompt1": {"1": 1}}, {"prompt1": {"2": 2}}, 
        {"prompt1": {"a": 'a'}}, {"prompt1": {"b": 'b'}}, {"prompt1": {"c": 'c'}}]
    sampler_train = MultiDatasetSampler([5], 6, consumed_samples=0, num_inference_per_prompt=2, shuffle=False, is_eval=False, data_parallel_rank=0, data_parallel_size=2)
    dataloader = RLHFDataLoader(datasets=[dataset1], sampler=sampler_train, collate_fn=collate_fn, num_inference_per_prompt=2, add_uid=True, data_parallel_rank=0, data_parallel_size=2, vllm_prompt_key="prompt1")
    data_iter = iter(dataloader)
    ground_truth = [
        {'prompt1': {'1': 1, 'uid': '0'}}, {'prompt1': {'1': 1, 'uid': '0'}}, 
        {'prompt1': {'2': 2, 'uid': '1'}}, {'prompt1': {'b': 'b', 'uid': '0'}}, 
        {'prompt1': {'b': 'b', 'uid': '0'}}, {'prompt1': {'c': 'c', 'uid': '1'}}, 
        {'prompt1': {'2': 2, 'uid': '0'}}, {'prompt1': {'2': 2, 'uid': '0'}}, 
        {'prompt1': {'a': 'a', 'uid': '1'}}]
    sequence = []
    for i in range(3):
        sequence.extend(next(data_iter))
    assert sequence == ground_truth

    sampler_train = MultiDatasetSampler([5], 6, consumed_samples=0, num_inference_per_prompt=2, shuffle=False, is_eval=False, data_parallel_rank=1, data_parallel_size=2)
    dataloader = RLHFDataLoader(datasets=[dataset1], sampler=sampler_train, collate_fn=collate_fn, num_inference_per_prompt=2, add_uid=True, data_parallel_rank=1, data_parallel_size=2, vllm_prompt_key="prompt1")
    data_iter = iter(dataloader)
    ground_truth = [
        {'prompt1': {'2': 2, 'uid': '1'}}, {'prompt1': {'a': 'a', 'uid': '2'}}, 
        {'prompt1': {'a': 'a', 'uid': '2'}}, {'prompt1': {'c': 'c', 'uid': '1'}}, 
        {'prompt1': {'1': 1, 'uid': '2'}}, {'prompt1': {'1': 1, 'uid': '2'}}, 
        {'prompt1': {'a': 'a', 'uid': '1'}}, {'prompt1': {'b': 'b', 'uid': '2'}}, 
        {'prompt1': {'b': 'b', 'uid': '2'}}]
    sequence = []
    for i in range(3):
        sequence.extend(next(data_iter))
    assert sequence == ground_truth

def drop_last0():

    # drop_last
    dataset = [1, 2, 3, 4, 5, 6, 7]
    sampler1 = MultiDatasetSampler([7], 6, [1], consumed_samples=0, num_inference_per_prompt=2, shuffle=True, is_eval=False, data_parallel_rank=0, data_parallel_size=2, drop_last="drop")
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler1, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches = []
    for i in range(5):
        batches.extend(next(data_iter))

    # drop_last_ckpt
    dataset = [1, 2, 3, 4, 5, 6, 7]
    sampler1 = MultiDatasetSampler([7], 6, [1], consumed_samples=6, num_inference_per_prompt=2, shuffle=True, is_eval=False, data_parallel_rank=0, data_parallel_size=2, drop_last="drop")
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler1, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches1 = []
    for i in range(5):
        batches1.extend(next(data_iter))
    assert batches1[:-3] == batches[3:]

    # drop_last
    sampler2 = MultiDatasetSampler([7], 6, [1], consumed_samples=0, num_inference_per_prompt=2, shuffle=True, is_eval=False, data_parallel_rank=1, data_parallel_size=2, drop_last="drop")
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler2, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches = []
    for i in range(5):
        batches.extend(next(data_iter))
    
    # drop_last_ckpt
    sampler2 = MultiDatasetSampler([7], 6, [1], consumed_samples=6, num_inference_per_prompt=2, shuffle=True, is_eval=False, data_parallel_rank=1, data_parallel_size=2, drop_last="drop")
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler2, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches2 = []
    for i in range(5):
        batches2.extend(next(data_iter))
    assert batches2[:-3] == batches[3:]

def drop_last1():

    # drop_last
    dataset = [1, 2, 3, 4, 5, 6, 7]
    sampler1 = MultiDatasetSampler([7], 6, [1], consumed_samples=0, num_inference_per_prompt=2, shuffle=True, is_eval=False, data_parallel_rank=0, data_parallel_size=2, drop_last="retain")
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler1, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches = []
    for i in range(5):
        batches.extend(next(data_iter))

    # drop_last_ckpt
    dataset = [1, 2, 3, 4, 5, 6, 7]
    sampler1 = MultiDatasetSampler([7], 6, [1], consumed_samples=6, num_inference_per_prompt=2, shuffle=True, is_eval=False, data_parallel_rank=0, data_parallel_size=2, drop_last="retain")
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler1, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches1 = []
    for i in range(5):
        batches1.extend(next(data_iter))
    assert batches1[:-1] == batches[3:]

    # drop_last
    sampler2 = MultiDatasetSampler([7], 6, [1], consumed_samples=0, num_inference_per_prompt=2, shuffle=True, is_eval=False, data_parallel_rank=1, data_parallel_size=2, drop_last="retain")
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler2, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches = []
    for i in range(5):
        batches.extend(next(data_iter))
    
    # drop_last_ckpt
    sampler2 = MultiDatasetSampler([7], 6, [1], consumed_samples=6, num_inference_per_prompt=2, shuffle=True, is_eval=False, data_parallel_rank=1, data_parallel_size=2, drop_last="retain")
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler2, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches2 = []
    for i in range(5):
        batches2.extend(next(data_iter))
    assert batches2[:-1] == batches[3:]

def rerank():
    # drop_last
    dataset = [1, 2, 3, 4, 5, 6, 7]
    sampler1 = MultiDatasetSampler([7], 6, [1], consumed_samples=0, num_inference_per_prompt=3, shuffle=False, is_eval=False, data_parallel_rank=0, data_parallel_size=2, drop_last="drop", data_rerank=True)
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler1, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches0 = []
    for i in range(5):
        batches0.extend(next(data_iter))

    # drop_last_ckpt
    dataset = [1, 2, 3, 4, 5, 6, 7]
    sampler1 = MultiDatasetSampler([7], 6, [1], consumed_samples=6, num_inference_per_prompt=3, shuffle=False, is_eval=False, data_parallel_rank=0, data_parallel_size=2, drop_last="drop", data_rerank=True)
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler1, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches1 = []
    for i in range(5):
        batches1.extend(next(data_iter))

    # drop_last
    sampler2 = MultiDatasetSampler([7], 6, [1], consumed_samples=0, num_inference_per_prompt=3, shuffle=False, is_eval=False, data_parallel_rank=1, data_parallel_size=2, drop_last="drop", data_rerank=True)
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler2, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches2 = []
    for i in range(5):
        batches2.extend(next(data_iter))
    
    # drop_last_ckpt
    sampler2 = MultiDatasetSampler([7], 6, [1], consumed_samples=6, num_inference_per_prompt=3, shuffle=False, is_eval=False, data_parallel_rank=1, data_parallel_size=2, drop_last="drop", data_rerank=True)
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler2, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches3 = []
    for i in range(5):
        batches3.extend(next(data_iter))

    assert batches0 == [1, 2, 1, 3, 4, 3, 5, 6, 5, 1, 2, 1, 3, 4, 3]
    assert batches1 == [3, 4, 3, 5, 6, 5, 1, 2, 1, 3, 4, 3, 5, 6, 5]
    assert batches2 == [2, 1, 2, 4, 3, 4, 6, 5, 6, 2, 1, 2, 4, 3, 4]
    assert batches3 == [4, 3, 4, 6, 5, 6, 2, 1, 2, 4, 3, 4, 6, 5, 6]

def dynamic_generation_batchsize():
    # drop_last
    dataset = [1, 2, 3, 4, 5, 6, 7]
    sampler1 = MultiDatasetSampler([7], 9, [1], consumed_samples=0, num_inference_per_prompt=3, shuffle=False, is_eval=False, data_parallel_rank=0, data_parallel_size=2, drop_last="drop", data_rerank=True)
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler1, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches0 = []
    for i in range(5):
        batches0.extend(next(data_iter))

    # drop_last_ckpt
    dataset = [1, 2, 3, 4, 5, 6, 7]
    sampler1 = MultiDatasetSampler([7], 9, [1], consumed_samples=9, num_inference_per_prompt=3, shuffle=False, is_eval=False, data_parallel_rank=0, data_parallel_size=2, drop_last="drop", data_rerank=True)
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler1, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches1 = []
    for i in range(5):
        batches1.extend(next(data_iter))

    # drop_last
    sampler2 = MultiDatasetSampler([7], 9, [1], consumed_samples=0, num_inference_per_prompt=3, shuffle=False, is_eval=False, data_parallel_rank=1, data_parallel_size=2, drop_last="drop", data_rerank=True)
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler2, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches2 = []
    for i in range(5):
        batches2.extend(next(data_iter))
    
    # drop_last_ckpt
    sampler2 = MultiDatasetSampler([7], 9, [1], consumed_samples=9, num_inference_per_prompt=3, shuffle=False, is_eval=False, data_parallel_rank=1, data_parallel_size=2, drop_last="drop", data_rerank=True)
    dataloader = RLHFDataLoader(datasets=[dataset], sampler=sampler2, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches3 = []
    for i in range(5):
        batches3.extend(next(data_iter))
    assert batches0 == [1, 2, 3, 1, 2, 4, 5, 6, 4, 5, 1, 2, 3, 1, 2, 4, 5, 6, 4, 5, 1, 2, 3, 1, 2]
    assert batches1 == [4, 5, 6, 4, 5, 1, 2, 3, 1, 2, 4, 5, 6, 4, 5, 1, 2, 3, 1, 2, 4, 5, 6, 4, 5]
    assert batches2 == [3, 1, 2, 3, 6, 4, 5, 6, 3, 1, 2, 3, 6, 4, 5, 6, 3, 1, 2, 3]
    assert batches3 == [6, 4, 5, 6, 3, 1, 2, 3, 6, 4, 5, 6, 3, 1, 2, 3, 6, 4, 5, 6]

if __name__ == "__main__":
    single_dataset()
    multiple_dataset()
    multi_replica()
    uid()
    drop_last0()
    drop_last1()
    rerank()
    dynamic_generation_batchsize()
