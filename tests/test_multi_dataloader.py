import os
from chatlearn.data.data import RLHFDataLoader


class RLHFSampler:
    def __init__(self, total_samples, consumed_samples, batch_size, base_seed=0):
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.batch_size = batch_size
        self.last_batch_size = self.total_samples % self.batch_size

        self.base_seed = base_seed
        self.curr_seed = base_seed

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.batch_size > 0

    def __len__(self):
        return self.total_samples

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch
    

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.batch_size == 0

        self.set_epoch(self.epoch)
        random_idx = [i for i in range(self.total_samples)]
        offset = current_epoch_samples
        idx_range = random_idx[offset:]

        batch = []
        # Last batch if not complete will be dropped.
        print(idx_range)
        for idx in idx_range:
            batch.append(idx)
            print(batch)
            if len(batch) == self.batch_size:
                self.consumed_samples += self.batch_size
                print(batch)
                yield batch
                batch = []

def test_case3():
    dataset1 = ['1', '2', '3']
    dataset2 = ['a', 'b']

    ratio = [2, 1]

    batch_sampler = RLHFSampler(3, batch_size=3, consumed_samples=3)

    data_iter = iter(RLHFDataLoader(datasets=[dataset1, dataset2], sampler=batch_sampler, consume_ratio=ratio, dataset_ratio=[3,2], num_inference_per_prompt=2))
    for i in range(20):
        print(next(data_iter))




if __name__ == "__main__":
    train_prompts = ['a', 'b', 'c', 'd', 'e', 'f']
    # test_case1()
    # test_case2()
    test_case3()

