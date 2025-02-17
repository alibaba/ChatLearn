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
"""data sampler."""

from itertools import chain
from chatlearn.utils import utils
import torch

class SingleDataSampler:
    """SingleDataSampler"""

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, dynamic_batch_size_flag=False, drop_last=False):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.remainder = (total_samples - consumed_samples) % data_parallel_size \
            if dynamic_batch_size_flag else 0
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size + self.remainder
        self.drop_last = drop_last
        self.data_parallel_size = data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_batch_size_plus = self.data_parallel_rank if self.data_parallel_rank < self.remainder else self.remainder
        start_idx = self.data_parallel_rank * self.micro_batch_size + start_batch_size_plus
        batch_size_plus = 1 if self.data_parallel_rank < self.remainder else 0
        batch_size = self.micro_batch_size + batch_size_plus
        end_idx = start_idx + batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            indices = utils.split_index(len(batch), self.data_parallel_size)
            start_idx, end_idx = indices[self.data_parallel_rank]
            yield batch[start_idx:end_idx]

class EpisodeDataSampler:
    """EpisodeDataSampler"""

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, sample_per_episode, drop_last=False):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last
        self.data_parallel_size = data_parallel_size
        if self.drop_last:
            last_samples = self.total_samples % self.micro_batch_times_data_parallel_size
            assert self.total_samples > last_samples, \
                'total_samples are not enough to perform drop_last!'
            self.total_samples -= last_samples

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        if self.consumed_samples >= self.total_samples:
            self.consumed_samples = self.consumed_samples % self.total_samples
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)
        self.episode_offset = 0
        self.sample_per_episode = sample_per_episode

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self, batch):
        indices = utils.split_index(len(batch), self.data_parallel_size)
        return indices[self.data_parallel_rank]

    def iter_internal(self, batch):
        # for cycle purpose
        if self.consumed_samples >= self.total_samples:
            self.consumed_samples = self.consumed_samples % self.total_samples
        for idx in chain(range(self.consumed_samples, self.total_samples), range(self.consumed_samples)):
            batch.append(idx)
            self.episode_offset += 1
            self.consumed_samples += 1
            if len(batch) == self.micro_batch_times_data_parallel_size or \
                    self.episode_offset == self.sample_per_episode:
                return True
        return False

    def __iter__(self):
        batch = []
        while True:
            # Last batch will be dropped if drop_last is set True
            batch_gen_flag = self.iter_internal(batch)
            # Check the last partial batch and see drop_last is set
            if len(batch) > 0 and not batch_gen_flag:
                # wrap it to sample_per_episode
                batch_gen_flag = self.iter_internal(batch)

            if batch_gen_flag:
                start_idx, end_idx = self.get_start_end_idx(batch)
                yield batch[start_idx:end_idx]
                batch = []

            if self.episode_offset == self.sample_per_episode:
                self.episode_offset = 0

class RLHFSingleSampler:

    def __init__(self, total_samples, consumed_samples, batch_size, shuffle=True, seed=0):
        self.actual_samples = total_samples // batch_size * batch_size
        self.consumed_samples = consumed_samples
        self.curr_epoch = self.consumed_samples // self.actual_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        """
            yield sampled indexes
        """
        while True:
            offset = self.consumed_samples % self.actual_samples
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.curr_epoch + self.seed)
                random_idx = torch.randperm(self.actual_samples, generator=g).tolist()
                # print(random_idx)
            else:
                assert len(random_idx) - offset >= self.batch_size
                random_idx = [i for i in range(self.actual_samples)]
            for i in range(offset, len(random_idx), self.batch_size):
                yield [random_idx[i + j] for j in range(self.batch_size)]
                self.consumed_samples += self.batch_size
            self.curr_epoch += 1


class MultiDatasetSampler:
    def __init__(
        self, 
        dataset_sizes, 
        batch_size, 
        data_ratio=None,
        consumed_samples=0,
        num_inference_per_prompt=1,
        shuffle=True,
        seeds=None,
        mix_batch=False,
        drop_last=True,
        init_shuffle_prompt=0
    ):

        self.dataset_sizes = dataset_sizes
        self.dataset_num = len(dataset_sizes)
        self.batch_size = batch_size
        self.consumed_samples = consumed_samples
        self.mix_batch = mix_batch
        self.num_inference_per_prompt = num_inference_per_prompt
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seeds = [0] * self.dataset_num if seeds is None else seeds

        if self.mix_batch:
            self.data_ratio = [1] * self.dataset_num if data_ratio is None else data_ratio
            self.batch_prop = self.cal_batch_proportion()
            # print(self.batch_prop)
            consumed_each = [self.consumed_samples * self.batch_prop[i] // sum(self.batch_prop) for i in range(self.dataset_num)]
            self.samplers = [RLHFSingleSampler(self.dataset_sizes[i], consumed_each[i], self.batch_prop[i], shuffle=self.shuffle, seed=self.seeds[i]) for i in range(self.dataset_num)] 

        assert init_shuffle_prompt == 0, "init_shuffle_prompt=1, 2 is not supported yet"
        assert self.drop_last or not self.mix_batch, "drop_last is not supported when mix_batch is True"

    def cal_batch_proportion(self):
        mixed_size = sum(self.data_ratio)
        remains = self.batch_size % mixed_size
        proportion = [r * (self.batch_size // mixed_size) for r in self.data_ratio]
        idx = 0
        while remains != 0:
            proportion[idx] += min(remains, self.data_ratio[idx])
            remains -= min(remains, self.data_ratio[idx])
            idx = (idx + 1) % self.dataset_num
        return proportion

    def merge_batches(self, batches):
        batches_with_idx = []
        for idx, b in enumerate(batches):
            batches_with_idx.append([(idx, data) for data in b])

        batch = []
        idx = 0
        while len(batch) != self.batch_size:
            batch.extend(batches_with_idx[idx][:min(self.data_ratio[idx], len(batches_with_idx[idx]))])
            batches_with_idx[idx] = batches_with_idx[idx][min(self.data_ratio[idx], len(batches_with_idx[idx])):]
            idx = (idx + 1) % self.dataset_num
        return batch
    
    def __iter__(self):
        if self.mix_batch:
            sampler_iters = [iter(sampler) for sampler in self.samplers]
            while True:
                batches = [next(it) for it in sampler_iters]
                merged_batch = self.merge_batches(batches)
                # duplicate
                duplicated_batch = []
                for data in merged_batch:
                    duplicated_batch.extend([data for i in range(self.num_inference_per_prompt)])
                yield duplicated_batch
        else:
            idxes = []
            for i in range(self.dataset_num):
                idxes.extend([(i, j) for j in range(self.dataset_sizes[i])])
            
            for i in range(0, len(idxes), self.batch_size):
                if self.drop_last and len(idxes) - i >= self.batch_size:
                    batch = [idxes[i + j] for j in range(self.batch_size)]
                elif not self.drop_last:
                    batch = [idxes[i + j] for j in range(min(self.batch_size, len(idxes) - i))]
                duplicated_batch = []
                for data in batch:
                    duplicated_batch.extend([data for i in range(self.num_inference_per_prompt)])
                yield duplicated_batch

