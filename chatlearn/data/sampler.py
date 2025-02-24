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

import torch
from chatlearn.utils import utils

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
    """RLHF sampler for a dataset.
    """

    def __init__(
        self,
        total_samples,
        consumed_samples,
        batch_size,
        shuffle=True,
        seed=0,
        num_inference_per_prompt=1,
        drop_last=False):

        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.actual_samples = total_samples * num_inference_per_prompt \
            // batch_size * batch_size if drop_last else total_samples * num_inference_per_prompt
        self.curr_epoch = consumed_samples // self.actual_samples
        self.shuffle = shuffle
        self.seed = seed
        self.num_inference_per_prompt = num_inference_per_prompt
        self.drop_last = drop_last
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.curr_epoch + self.seed)
            self.random_idx = torch.randperm(self.total_samples, generator=g).tolist()
        else:
            self.random_idx = list(range(self.total_samples))
        self.duplicated_random_idx = []
        for ridx in self.random_idx:
            self.duplicated_random_idx.extend([ridx] * self.num_inference_per_prompt)
        self.offset = consumed_samples % self.actual_samples

    def get_next(self, num):
        batch = []
        if self.drop_last:
            assert num <= self.total_samples * self.num_inference_per_prompt, \
                'drop_last does not support batch_size larger than dataset size'
            if num > self.total_samples * self.num_inference_per_prompt - self.offset:
                self.offset = 0
                self.curr_epoch += 1
                if self.shuffle:
                    g = torch.Generator()
                    g.manual_seed(self.curr_epoch + self.seed)
                    self.random_idx = torch.randperm(self.total_samples, generator=g).tolist()
                    self.duplicated_random_idx = []
                    for ridx in self.random_idx:
                        self.duplicated_random_idx.extend([ridx] * self.num_inference_per_prompt)
            batch.extend(self.duplicated_random_idx[self.offset : self.offset + num])
            self.offset = self.offset + num
            return batch
        else:
            while num >= self.total_samples * self.num_inference_per_prompt - self.offset:
                batch.extend(self.duplicated_random_idx[self.offset : self.total_samples * self.num_inference_per_prompt])
                num -= self.total_samples * self.num_inference_per_prompt - self.offset
                self.offset = 0
                self.curr_epoch += 1
                if self.shuffle:
                    g = torch.Generator()
                    g.manual_seed(self.curr_epoch + self.seed)
                    self.random_idx = torch.randperm(self.total_samples, generator=g).tolist()
                    self.duplicated_random_idx = []
                    for ridx in self.random_idx:
                        self.duplicated_random_idx.extend([ridx] * self.num_inference_per_prompt)

            batch.extend(self.duplicated_random_idx[self.offset : self.offset + num])
            self.offset = self.offset + num
            return batch

class MultiDatasetSampler:
    """RLHF sampler for multiple datasets.
    """
    def __init__(
        self,
        dataset_sizes,
        batch_size,
        data_ratio=None,
        consumed_samples=0,
        num_inference_per_prompt=1,
        shuffle=True,
        seed=None,
        is_eval=False,
        init_shuffle_prompt=0,
        data_parallel_rank=0,
        data_parallel_size=1,
        dynamic_batch_size_flag=False,
        drop_last=False
    ):
        self.remainder = (sum(dataset_sizes) - consumed_samples) % data_parallel_size \
            if dynamic_batch_size_flag else 0
        self.dataset_sizes = dataset_sizes
        self.dataset_num = len(dataset_sizes)
        self.micro_batch_size = batch_size
        self.data_parallel_size = data_parallel_size
        self.data_parallel_rank = data_parallel_rank
        self.batch_size = self.micro_batch_size * data_parallel_size
        self.consumed_samples = consumed_samples
        self.is_eval = is_eval
        self.num_inference_per_prompt = num_inference_per_prompt
        self.shuffle = shuffle
        self.seeds = [0] * self.dataset_num if seed is None else [seed] * self.dataset_num
        self.drop_last = drop_last

        if not self.is_eval and data_ratio is not None:
            assert len(data_ratio) == self.dataset_num
        assert init_shuffle_prompt == 0, "init_shuffle_prompt=1, 2 is not supported yet"
        assert self.consumed_samples % self.batch_size == 0, "consumed samples must be integer multiple of micro_batch_size times data_parallel_size"
        assert self.consumed_samples % self.num_inference_per_prompt == 0, "consumed samples must be integer multiple of num_inference_per_prompt"

        if not self.is_eval:
            if data_ratio is None:
                data_ratio = [1] * self.dataset_num
            elif isinstance(data_ratio, int):
                data_ratio = [data_ratio] * self.dataset_num
            elif isinstance(data_ratio, list):
                assert len(data_ratio) == self.dataset_num, (
                    "expect data_ratio to be a list with the same length as the number of datasets, "
                    f"got {len(data_ratio)} and {self.dataset_num}."
                )
            else:
                raise TypeError(f"unexpected data_ratio type {type(data_ratio)}, expect int or List.")

            self.data_ratio = [self.num_inference_per_prompt] * self.dataset_num if data_ratio is None \
                else [r * self.num_inference_per_prompt for r in data_ratio]
            consumed_each, self.dataset_remains = self.cal_consumed_each(self.consumed_samples, self.data_ratio)
            self.samplers = [
                RLHFSingleSampler(
                    self.dataset_sizes[i],
                    consumed_each[i],
                    shuffle=self.shuffle,
                    seed=self.seeds[i],
                    num_inference_per_prompt=num_inference_per_prompt,
                    drop_last=self.drop_last,
                    batch_size=self.batch_size
                )
                for i in range(self.dataset_num)
            ]

    def cal_consumed_each(self, consumed_samples, data_ratio):
        multiples = consumed_samples // sum(data_ratio)
        consumed_each = [r * multiples for r in data_ratio]
        remains = consumed_samples % sum(data_ratio)
        i = 0
        dataset_remains = data_ratio[:]
        while remains != 0:
            if i == 0:
                dataset_remains = data_ratio[:]
            dataset_remains[i] -= min(remains, data_ratio[i])
            consumed_each[i] += min(remains, data_ratio[i])
            remains -= min(remains, data_ratio[i])
            i = (i + 1) % self.dataset_num

        return consumed_each, dataset_remains

    def __iter__(self):
        if self.is_eval:
            idxes = []
            for i in range(self.dataset_num):
                idxes.extend([(i, j) for j in range(self.dataset_sizes[i])])

            if self.data_parallel_rank >= self.remainder:
                batch_size_list = [self.micro_batch_size + 1] * self.remainder + \
                    [self.micro_batch_size] * (self.data_parallel_size - self.remainder)
            else:
                batch_size_list = [self.micro_batch_size] * self.remainder + \
                    [self.micro_batch_size - 1] * (self.data_parallel_size - self.remainder)
            left = sum(batch_size_list[:self.data_parallel_rank])
            right = sum(batch_size_list[:self.data_parallel_rank + 1])
            batch = idxes[left : right]
            duplicated_batch = []
            for data in batch:
                duplicated_batch.extend([data for i in range(self.num_inference_per_prompt)])
            yield duplicated_batch
        else:
            if self.dataset_num == 1 and self.drop_last:
                while True:
                    batch = self.samplers[0].get_next(self.batch_size)
                    batch_idxes = [(0, batch[i]) for i in range(self.batch_size)]
                    batch_idxes = batch_idxes[self.data_parallel_rank * self.micro_batch_size : (self.data_parallel_rank + 1) * self.micro_batch_size]
                    yield batch_idxes
            else:
                while True:
                    batch_idxes = []
                    dataset_id = 0
                    while len(batch_idxes) != self.batch_size:
                        data_num = min(self.dataset_remains[dataset_id], self.batch_size - len(batch_idxes))
                        self.dataset_remains[dataset_id] -= data_num
                        batch = self.samplers[dataset_id].get_next(data_num)
                        batch_idxes.extend([(dataset_id, batch[i]) for i in range(data_num)])
                        dataset_id = (dataset_id + 1) % self.dataset_num
                        if self.dataset_remains == [0] * self.dataset_num:
                            self.dataset_remains = self.data_ratio[:]

                    batch_idxes = batch_idxes[self.data_parallel_rank * self.micro_batch_size : (self.data_parallel_rank + 1) * self.micro_batch_size]
                    yield batch_idxes
