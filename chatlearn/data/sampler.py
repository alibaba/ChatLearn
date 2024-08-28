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
            if len(batch) > 0 and not self.drop_last and not batch_gen_flag:
                # wrap it to sample_per_episode
                batch_gen_flag = self.iter_internal(batch)

            if batch_gen_flag:
                start_idx, end_idx = self.get_start_end_idx(batch)
                yield batch[start_idx:end_idx]
                batch = []

            if self.episode_offset == self.sample_per_episode:
                self.episode_offset = 0
