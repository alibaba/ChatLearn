import os
from chatlearn.data.data import RLHFDataLoader
from chatlearn.data.sampler import MultiDatasetSampler
from itertools import cycle

def collate_fn(batch):
    return batch

def single_dataset():
    # evaluation
    dataset = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    sampler_eval = MultiDatasetSampler([len(dataset)], 3, consumed_samples=0, num_inference_per_prompt=2, shuffle=False, is_eval=True)
    dataloader = RLHFDataLoader([dataset], sampler_eval, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    for i in range(5):
        batches = []
        for j in range(5):
            batches.append(next(data_iter))
        ground_truth = [['a', 'a', 'b', 'b', 'c', 'c'], ['d', 'd', 'e', 'e', 'f', 'f'], ['g', 'g', 'h', 'h', 'i', 'i'], ['a', 'a', 'b', 'b', 'c', 'c'], ['d', 'd', 'e', 'e', 'f', 'f']]
        assert batches == ground_truth
        data_iter = cycle(iter(dataloader))

    # training
    dataset1 = [1, 2]
    sampler_train = MultiDatasetSampler([2], 3, [1], consumed_samples=0, num_inference_per_prompt=2, shuffle=False, is_eval=False)
    dataloader = RLHFDataLoader(datasets=[dataset1], sampler=sampler_train, collate_fn=collate_fn)
    data_iter = cycle(iter(dataloader))
    batches = []
    for i in range(5):
        batches.append(next(data_iter))
    ground_truth = [[1,1,2,2,1,1], [2,2,1,1,2,2], [1,1,2,2,1,1], [2,2,1,1,2,2], [1,1,2,2,1,1]]
    assert batches == ground_truth

    # checkpoint
    dataset1 = [1, 2]
    sampler_train = MultiDatasetSampler([2], 3, [1], consumed_samples=3, num_inference_per_prompt=2, shuffle=False, is_eval=False)
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
    sampler_eval = MultiDatasetSampler([5, 9], 3, consumed_samples=0, num_inference_per_prompt=2, shuffle=False, is_eval=True)
    dataloader = RLHFDataLoader([dataset1, dataset2], sampler_eval, collate_fn=collate_fn)
    ground_truth = [[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 'a', 'a'], ['b', 'b', 'c', 'c', 'd', 'd'], ['e', 'e', 'f', 'f', 'g', 'g']]
    data_iter = cycle(iter(dataloader))
    for i in range(5):
        batches = []
        for j in range(4):
            batches.append(next(data_iter))
        data_iter = cycle(iter(dataloader)) # reset
        assert batches == ground_truth

    # training
    sampler_train = MultiDatasetSampler([5, 9], 5, [1, 3], consumed_samples=0, num_inference_per_prompt=2, shuffle=True, is_eval=False)
    dataloader = RLHFDataLoader(datasets=[dataset1, dataset2], sampler=sampler_train, collate_fn=collate_fn)
    data_iter = iter(dataloader)
    sequence = []
    for i in range(10):
        sequence.extend(next(data_iter))
    # num_inference_per_prompt
    assert len(sequence) == 10 * 2 * 5

    for i in range(0, len(sequence) // 2, 2):
        assert sequence[i] == sequence[i + 1]

    # data checkpoint
    sampler_train = MultiDatasetSampler([5, 9], 5, [1, 3], consumed_samples=15, num_inference_per_prompt=2, shuffle=True, is_eval=False)
    dataloader = RLHFDataLoader(datasets=[dataset1, dataset2], sampler=sampler_train, collate_fn=collate_fn)
    data_iter = iter(dataloader)
    new_sequence = []
    for i in range(10):
        new_sequence.extend(next(data_iter))
    assert sequence[30:] == new_sequence[:len(sequence) - 30]

def multi_replica():
    # evaluation
    dataset1 = [1, 2, 3, 4, 5]
    dataset2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    sampler_eval = MultiDatasetSampler([5, 9], 3, consumed_samples=0, num_inference_per_prompt=2, shuffle=False, is_eval=True, data_parallel_rank=1, data_parallel_size=2)
    dataloader = RLHFDataLoader([dataset1, dataset2], sampler_eval, collate_fn=collate_fn)
    ground_truth = [[4, 4, 5, 5, 'a', 'a'], ['e', 'e', 'f', 'f', 'g', 'g']]
    data_iter = cycle(iter(dataloader))
    for i in range(5):
        batches = []
        for j in range(2):
            batches.append(next(data_iter))
        # print(batches)
        data_iter = cycle(iter(dataloader)) # reset
        assert batches == ground_truth

    # training
    sampler_train = MultiDatasetSampler([5, 9], 5, [1, 3], consumed_samples=0, num_inference_per_prompt=2, shuffle=False, is_eval=False, data_parallel_rank=0, data_parallel_size=2)
    dataloader = RLHFDataLoader(datasets=[dataset1, dataset2], sampler=sampler_train, collate_fn=collate_fn)
    data_iter = iter(dataloader)
    sequence = []
    ground_truth = [1, 1, 'a', 'a', 'b', 'b', 'c', 'c', 2, 2, 'h', 'h', 'i', 'i', 4, 4, 'a', 'a', 'b', 'b', 1, 1, 'g', 'g', 'h', 'h', 'i', 'i', 2, 2]
    for i in range(3):
        sequence.extend(next(data_iter))
    assert sequence == ground_truth

    sampler_train = MultiDatasetSampler([5, 9], 5, [1, 3], consumed_samples=0, num_inference_per_prompt=2, shuffle=False, is_eval=False, data_parallel_rank=1, data_parallel_size=2)
    dataloader = RLHFDataLoader(datasets=[dataset1, dataset2], sampler=sampler_train, collate_fn=collate_fn)
    data_iter = iter(dataloader)
    sequence = []
    ground_truth = ['d', 'd', 'e', 'e', 'f', 'f', 3, 3, 'g', 'g', 'c', 'c', 5, 5, 'd', 'd', 'e', 'e', 'f', 'f', 'a', 'a', 'b', 'b', 'c', 'c', 3, 3, 'd', 'd']
    for i in range(3):
        sequence.extend(next(data_iter))
    assert sequence == ground_truth



if __name__ == "__main__":
    single_dataset()
    multiple_dataset()
    multi_replica()