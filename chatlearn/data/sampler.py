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
        drop_last="cycle"):

        """
        drop_last:
            "drop": drop last
            "retain": return remaining samples
            "cycle": loop back to the beginning
        """

        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples

        self.actual_samples = total_samples // batch_size * batch_size if self.drop_last == "drop" else total_samples
        self.offset = consumed_samples % self.actual_samples
        self.curr_epoch = consumed_samples // self.actual_samples
        self.update_random_idxs()

    def update_random_idxs(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.curr_epoch + self.seed)
            self.random_idx = torch.randperm(self.total_samples, generator=g).tolist()
        else:
            self.random_idx = list(range(self.total_samples))

    def get_next(self, num):
        batch = []
        if self.drop_last == "drop":
            assert num <= self.total_samples, 'drop mode does not support batch_size larger than dataset size'
            if num > self.total_samples - self.offset:
                self.offset = 0
                self.curr_epoch += 1
                self.update_random_idxs()
            batch.extend(self.random_idx[self.offset : self.offset + num])
            self.offset = self.offset + num
            return batch
        elif self.drop_last == "retain":
            if num > self.total_samples - self.offset:
                batch.extend(self.random_idx[self.offset : self.total_samples])
                self.offset = 0
                self.curr_epoch += 1
                self.update_random_idxs()
            else:
                batch.extend(self.random_idx[self.offset : self.offset + num])
                self.offset = self.offset + num
            return batch
        else:
            while num >= self.total_samples - self.offset:
                batch.extend(self.random_idx[self.offset : self.total_samples])
                num -= (self.total_samples - self.offset)
                self.offset = 0
                self.curr_epoch += 1
                self.update_random_idxs()

            batch.extend(self.random_idx[self.offset : self.offset + num])
            self.offset = self.offset + num
            return batch

class MultiDatasetSampler:
    """RLHF sampler for multiple datasets.
    """
    def __init__(
        self,
        dataset_sizes,
        sample_per_episode,
        data_ratio=None,
        consumed_samples=0,
        num_inference_per_prompt=1,
        shuffle=True,
        seed=None,
        is_eval=False,
        init_shuffle_prompt=0,
        data_parallel_rank=0,
        data_parallel_size=1,
        drop_last="cycle",
        data_rerank=False
    ):
        """
        drop_last:
            "drop": drop last
            "retain": return remaining samples
            "cycle": loop back to the beginning
        """
        self.dataset_sizes = dataset_sizes
        self.dataset_num = len(dataset_sizes)
        self.data_parallel_size = data_parallel_size
        self.data_parallel_rank = data_parallel_rank
        self.sample_per_episode = sample_per_episode
        self.consumed_samples = consumed_samples
        self.is_eval = is_eval
        self.num_inference_per_prompt = num_inference_per_prompt
        self.shuffle = shuffle
        self.seeds = [0] * self.dataset_num if seed is None else [seed] * self.dataset_num
        if self.dataset_num == 1:
            self.drop_last = drop_last
        else:
            self.drop_last = "cycle"
        self.data_rerank = data_rerank

        assert init_shuffle_prompt == 0, "init_shuffle_prompt=1, 2 is not supported yet"
        assert self.consumed_samples % self.num_inference_per_prompt == 0, "consumed samples must be integer multiple of num_inference_per_prompt"
        assert self.sample_per_episode % self.num_inference_per_prompt == 0, "sample_per_episode must be integer multiple of num_inference_per_prompt"

        # need list[int] in length self.dataset_num
        if not self.is_eval:
            # [] or None
            if not data_ratio:
                self.data_ratio = [1] * self.dataset_num
            elif isinstance(data_ratio, int):
                self.data_ratio = [data_ratio] * self.dataset_num
            elif isinstance(data_ratio, list):
                assert len(data_ratio) == self.dataset_num, (
                    "expect data_ratio to be a list with the same length as the number of datasets, "
                    f"got {len(data_ratio)} and {self.dataset_num}."
                )
                self.data_ratio = data_ratio
            else:
                raise TypeError(f"unexpected data_ratio type {type(data_ratio)}, expect int or List.")

            consumed_each, self.dataset_remains = self.cal_consumed_each(self.consumed_samples // self.num_inference_per_prompt, self.data_ratio)
            self.samplers = [
                RLHFSingleSampler(
                    self.dataset_sizes[i],
                    consumed_each[i],
                    batch_size=self.sample_per_episode // self.num_inference_per_prompt if self.dataset_num == 1 else self.data_ratio[i],
                    shuffle=self.shuffle,
                    seed=self.seeds[i],
                    drop_last=self.drop_last
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

    def repeat(self, data, interleave=False):
        if interleave:
            res = []
            for d in data:
                res.extend([d] * self.num_inference_per_prompt)
            return res
        else:
            return data * self.num_inference_per_prompt

    def __iter__(self):
        self.remainder = self.sample_per_episode % self.data_parallel_size
        batch_size_list = [self.sample_per_episode // self.data_parallel_size + 1] * self.remainder + \
            [self.sample_per_episode // self.data_parallel_size] * (self.data_parallel_size - self.remainder)
        start, end = sum(batch_size_list[:self.data_parallel_rank]), sum(batch_size_list[:self.data_parallel_rank + 1])

        if self.is_eval:
            assert self.sample_per_episode <= sum(self.dataset_sizes), "eval dataset size must be larger than sample_per_episode"
            idxes = []
            for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
                idxes.extend([(dataset_idx, j, (len(idxes) + j)) for j in range(dataset_size)])
            for i in range(0, len(idxes), self.sample_per_episode):
                episode_samples = idxes[i : i + self.sample_per_episode]
                yield episode_samples[start : end]
        else:
            num_samples = self.sample_per_episode // self.num_inference_per_prompt
            while True:
                if self.drop_last == "drop":
                    batch = self.samplers[0].get_next(num_samples)
                    batch_idxes = [(0, batch[i], i) for i in range(len(batch))]
                    batch_idxes = self.repeat(batch_idxes, interleave=not self.data_rerank)
                    batch_idxes = batch_idxes[start : end]
                elif self.drop_last == "retain":
                    batch = self.samplers[0].get_next(num_samples)
                    batch_idxes = [(0, batch[i], i) for i in range(len(batch))]
                    batch_idxes = self.repeat(batch_idxes, interleave=not self.data_rerank)
                    new_batch_size_list = [len(batch_idxes) // self.data_parallel_size] * self.data_parallel_size
                    for i in range(len(batch_idxes) % self.data_parallel_size):
                        new_batch_size_list[i] += 1
                    batch_idxes = \
                        batch_idxes[sum(new_batch_size_list[:self.data_parallel_rank]) : sum(new_batch_size_list[:self.data_parallel_rank + 1])]
                else:
                    batch_idxes = []
                    dataset_id = 0
                    while len(batch_idxes) != num_samples:
                        data_num = min(self.dataset_remains[dataset_id], num_samples - len(batch_idxes))
                        self.dataset_remains[dataset_id] -= data_num
                        batch = self.samplers[dataset_id].get_next(data_num)
                        batch_idxes.extend([(dataset_id, batch[i], (len(batch_idxes) + i)) for i in range(data_num)])
                        dataset_id = (dataset_id + 1) % self.dataset_num
                        if self.dataset_remains == [0] * self.dataset_num:
                            self.dataset_remains = self.data_ratio[:]
                    batch_idxes = self.repeat(batch_idxes, interleave=not self.data_rerank)
                    batch_idxes = batch_idxes[start : end]
                yield batch_idxes
