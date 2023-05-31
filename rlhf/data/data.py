# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
"""data processing."""

import math
import random
from itertools import cycle

import ray
import torch
from torch.nn.utils.rnn import pad_sequence

from rlhf.utils import future


def get_iter_keys(data):
    """
    get iterator keys
    """
    if isinstance(data, (list, tuple)):
        return range(len(data))
    if isinstance(data, dict):
        return data.keys()
    raise ValueError(f"Only list or dict type is accepted, but got type {type(data)}")


def create_from_type(data):
    """
    create collection from data type
    """
    if isinstance(data, (list, tuple)):
        return [None] * len(data)
    return type(data)()


def batching(tensors, padding_value=0.0, padding_type="right"):
    """
    batch tensors
    """
    if isinstance(tensors[0], torch.Tensor):
        if tensors[0].dim() == 0:
            return torch.stack(tensors)
        if padding_type == "right":
            return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
        return pad_sequence([elem.flip(0) for elem in tensors],
                            padding_value=padding_value,
                            batch_first=True).flip(1)
    batch = create_from_type(tensors[0])
    batch_size = len(tensors)
    for key in get_iter_keys(tensors[0]):
        pad = padding_value.get(key, 0.0) if isinstance(padding_value,
                                                        dict) else padding_value
        ptype = padding_type.get(key, "right") if isinstance(padding_type, dict) else padding_type
        batched = [tensors[j][key] for j in range(batch_size)]
        if isinstance(batched[0], torch.Tensor):
            batched = batching(batched, pad, ptype)
        batch[key] = batched
    return batch


def split_batch(batch):
    """
    split batch into samples
    """
    assert isinstance(batch, (list, tuple, dict)), \
        "batch type {} is not supported".format(type(batch))
    samples = []
    if isinstance(batch, (list, tuple)):
        batch_size = len(batch[0])
        keys = range(len(batch))
    else:
        batch_size = len(next(iter(batch.values())))
        keys = batch.keys()

    for batch_index in range(batch_size):
        if isinstance(batch, (list, tuple)):
            sample = [batch[key][batch_index] for key in keys]
        else:
            sample = {key: batch[key][batch_index] for key in keys}
        samples.append(sample)

    return samples


@ray.remote
class StreamDataset():
    """dataset built from queues"""

    def __init__(self, queue, total_samples, batch_size, padding_config=None, cache=False):
        """
        Args:
            total_samples: if total_samples is 0, then this is dynamic size Dataset
        """
        self.queue = queue
        self.total_samples = total_samples
        if self.total_samples == 0:
            self._dynamic_dataset = True
            self._num_batches = 0
        else:
            self._dynamic_dataset = False
            self._num_batches = math.ceil(total_samples / batch_size)
        self.batch_size = batch_size
        self.produce_index = 0
        self.cache = cache
        self.relay_buffer = []
        self.iter = self.__iter__()
        self._padding_config = padding_config if padding_config is not None else {}
        self._padding_value = {key: value["padding_value"] for key, value in self._padding_config.items()}
        self._padding_type = {key: value["padding_type"] for key, value in self._padding_config.items()}
        self._has_next = True

    def shuffle(self):
        """
        shuffle relay buffer
        """
        random.shuffle(self.relay_buffer)
        self.iter = self.__iter__() # pylint: disable=unnecessary-dunder-call
        self._has_next = True

    def __iter__(self):
        if self._dynamic_dataset:
            return self.iter_dynamic()
        return self.iter_fixed()

    def iter_fixed(self):
        """
        iteration with fixed batch size
        """
        self.produce_index = 0
        if len(self.relay_buffer) == self.total_samples:
            self.cache = False
        batch_count = 0
        while self.produce_index < self.total_samples:
            # read from cache
            if len(self.relay_buffer) < self.total_samples:
                while len(self.relay_buffer) < self.total_samples and \
                    (len(self.relay_buffer) - self.produce_index) < self.batch_size:
                    if self.queue.qsize() == 0:
                        raise ValueError("WARN: data queue is empty")
                    # get from queue
                    data = self.queue.get()
                    merged_data = {}
                    for item in data:
                        local_data = future.get(item)
                        merged_data.update(local_data)
                    samples = split_batch(merged_data)
                    self.relay_buffer += samples
            start_index = self.produce_index
            end_index = min(self.produce_index + self.batch_size, len(self.relay_buffer))
            data_to_batch = self.relay_buffer[start_index: end_index]
            if len(data_to_batch) < self.batch_size:
                data_to_batch += self.relay_buffer[:self.batch_size - len(data_to_batch)]
            batched_data = batching(data_to_batch, self._padding_value, self._padding_type)
            yield batched_data
            batch_count += 1
            self.produce_index += len(data_to_batch)
        assert batch_count == self._num_batches
        assert self.produce_index == len(self.relay_buffer)

    def iter_dynamic(self):
        """
        iteration with dynamic batch size
        """
        self.produce_index = 0
        if self.total_samples > 0:
            return self.iter_fixed()
        batch_count = 0

        while self.queue.qsize() > 0:
            while self.queue.qsize() > 0 and \
                (len(self.relay_buffer) - self.produce_index) < self.batch_size:
                # get from queue
                data = self.queue.get()
                merged_data = {}
                for item in data:
                    local_data = future.get(item)
                    merged_data.update(local_data)
                samples = split_batch(merged_data)
                self.relay_buffer += samples
            start_index = self.produce_index
            end_index = min(self.produce_index + self.batch_size, len(self.relay_buffer))
            data_to_batch = self.relay_buffer[start_index: end_index]
            if len(data_to_batch) < self.batch_size:
                data_to_batch += self.relay_buffer[:self.batch_size - len(data_to_batch)]
            batched_data = batching(data_to_batch, self._padding_value, self._padding_type)
            yield batched_data
            batch_count += 1
            self.produce_index += len(data_to_batch)
        self.total_samples = len(self.relay_buffer)
        self.num_batches = batch_count

    def next(self):
        """get next batch"""
        try:
            data = next(self.iter)
            return data
        except StopIteration:
            self._has_next = False
            return None

    def has_next(self):
        """
        has next batch
        """
        return self._has_next


class RLHFDataLoader:
    """
    RLHF data loader
    """

    def __init__(self, dataset, batch_size):
        """generate prompts data loader"""
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_iter = cycle(iter(self.dataset))

    def __iter__(self):
        batch_data = []
        for _, item in enumerate(self.data_iter):
            batch_data.append(item)
            if len(batch_data) == self.batch_size:
                batched = batching(batch_data)
                yield batched
                batch_data = []
        if len(batch_data) > 0:
            batched = batching(batch_data)
            yield batched
