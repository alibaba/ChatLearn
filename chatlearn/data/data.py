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
"""data processing."""

import math
import random
import copy
import os
import json
from typing import List, Dict, Union, Tuple

import ray
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate

from chatlearn.utils import future
from chatlearn.utils.constant import REF_LIST
from chatlearn.utils.utils import map_reduce_metrics

def read_data_path_list(data_path_list: List[str], mode: str = "jsonl"):
    data = []
    for data_path in data_path_list:
        if mode == "json":
            with open(data_path, 'r', encoding='utf-8') as f:
                data.extend(json.load(f))
        elif mode == "jsonl":
            with open(data_path, 'r', encoding='utf-8') as f:
                data.extend([json.loads(line) for line in f])
    return data


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


def batching(batches: List[Union[torch.Tensor, Dict, List, Tuple]]):
    """
    batch batches
    """
    if isinstance(batches[0], torch.Tensor):
        if batches[0].dim() == 0:
            return torch.stack(batches)
        return pad_sequence(batches, batch_first=True, padding_value=0.0)
    batch = create_from_type(batches[0])
    batch_size = len(batches)
    for key in get_iter_keys(batches[0]):
        batched = [batches[j][key] for j in range(batch_size)]
        if isinstance(batched[0], torch.Tensor):
            batched = batching(batched)
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
class StreamDataset:
    """dataset built from queues"""

    def __init__(self, data_loader_type, num_minibsz, micro_batch_size, max_replay_episode=0, replay_episode_offset=0):
        """
        Args:
            data_loader_type: fixed or dynamic
        """
        if data_loader_type == "fixed":
            self._dynamic_dataset = False
        else:
            self._dynamic_dataset = True
        self.num_minibsz = num_minibsz
        self.dp_size = None
        self.batch_size = micro_batch_size

        if max_replay_episode < 0:
            max_replay_episode = math.inf
        self._max_replay_episode = max_replay_episode
        self._replay_episode_offset = replay_episode_offset
        self._episode_replay_buffers = []
        self.replay_sample_manager = None

    def shuffle(self):
        """
        shuffle replay buffer
        """
        self.replay_buffer.shuffle()
        self.iter = self.__iter__() # pylint: disable=unnecessary-dunder-call
        self._has_next = True

    def __iter__(self):
        """
        entrypoint
        """
        if self._dynamic_dataset and not self._read_data_complete:
            return self.iter_dynamic()
        return self.iter_fixed()

    def _get_batch(self, start_index: int):
        end_index = min(start_index + self.batch_size, len(self.replay_buffer))
        data_to_batch = self.replay_buffer.get_samples(start_index, end_index)
        if len(data_to_batch) < self.batch_size:
            data_to_batch += self.replay_buffer.get_samples(0, self.batch_size - len(data_to_batch))
        return data_to_batch

    def iter_fixed(self):
        """
        iteration with fixed batch size
        return a group of data with 
        """
        produce_index = 0
        batch_count = 0
        assert self.dp_size is not None, "dp_size must be set before data fetching"
        self.batch_size = self._total_samples // self.num_minibsz // self.dp_size

        while produce_index < self._total_samples:
            # read from cache
            if len(self.replay_buffer) < self._total_samples:
                while len(self.replay_buffer) < self._total_samples and \
                    (len(self.replay_buffer) - produce_index) < self.batch_size:
                    self.replay_buffer.add_raw_batch()
            batched_data = self._get_batch(produce_index)
            yield batched_data
            batch_count += 1
            produce_index += self.batch_size

        assert batch_count == math.ceil(self._total_samples / self.batch_size)
        assert produce_index >= len(self.replay_buffer), \
               f"produce_index: {produce_index} < len(self.replay_buffer) {len(self.replay_buffer)}"

    def iter_dynamic(self):
        """
        iteration with dynamic batch size
        """
        produce_index = 0
        if self._read_data_complete:
            return self.iter_fixed()
        batch_count = 0

        while self.replay_buffer.queue_not_empty():
            while self.replay_buffer.queue_not_empty() and \
                (len(self.replay_buffer) - produce_index) < self.batch_size:
                # get from queue
                self.replay_buffer.add_raw_batch()
            batched_data = self._get_batch(produce_index)
            yield batched_data
            batch_count += 1
            produce_index += self.batch_size
        self._read_data_complete = True
        assert len(self.replay_buffer) == self._total_samples
        self._num_batches = batch_count

    def next(self):
        """get next batch"""
        return next(self.iter)

    def has_next(self):
        """
        has next batch
        """
        return self._has_next

    def set_dataset(self, queue, episode_id, replay_sample_manager=None, sample_per_episode=-1):
        replay_buffer = EpisodeReplayBuffer(episode_id, queue=queue)
        if self._max_replay_episode > 0 and episode_id >= self._replay_episode_offset:
            self._episode_replay_buffers.append(replay_buffer)
            if len(self._episode_replay_buffers) > self._max_replay_episode:
                old_buffer = self._episode_replay_buffers.pop(0)
                del old_buffer

            # this function will sync until all data computing finished,
            # which will block training until environment rollout finished.
            if os.getenv("SKIP_GENERATION", None) is None:
                replay_buffer.sync()
            if replay_sample_manager is None:
                raise Exception("default replay sample function is not currently supported")

            self.replay_sample_manager = replay_sample_manager
            buffer = self.replay_sample_manager(self._episode_replay_buffers)
            self.replay_buffer = EpisodeReplayBuffer(episode_id, buffer=buffer)
            self._total_samples = len(self.replay_buffer)
            self._read_data_complete = True
        else:
            num_rollout_batches = queue.qsize()
            self.replay_buffer = replay_buffer
            self.replay_buffer.add_raw_batch()
            assert sample_per_episode != -1, "In fixed batch size, you must set sample_per_episode for StreamDataset."
            self._total_samples = sample_per_episode
            self._read_data_complete = num_rollout_batches <= 1
        self.iter = iter(self)
        self._has_next = True

    def set_dp_size(self, dp_size:int):
        self.dp_size = dp_size

    def episode_replay_buffers(self):
        return self._episode_replay_buffers

    def num_iteration(self):
        return self.num_minibsz

    def total_samples(self):
        return self._total_samples

    def batch_per_episode(self):
        return math.ceil(self._total_samples / self.batch_size)

    def get_and_clear_metrics(self):
        # TODO: deal with situation that replay_sample_manager is None
        try:
            return self.replay_sample_manager.get_and_clear_metrics()
        except Exception:
            return "no replay", {}

class EpisodeReplayBuffer:
    """EpisodeReplayBuffer"""

    def __init__(self, episode_id, queue=None, buffer=None):
        self._episode_id = episode_id
        assert (queue is None or buffer is None) and (queue is not None or buffer is not None)
        if buffer is not None:
            assert queue is None
            self._buffer = buffer
        else:
            assert queue is not None
            self._buffer = []
        self.queue = queue
        self._rollout_batch_size = -1

    def add_raw_batch(self):
        if self.queue.qsize() == 0:
            raise ValueError("WARN: data queue is empty")
        # get from queue
        data = self.queue.get()
        samples = []
        for item in data:
            local_data = future.get(item)
            if isinstance(local_data, list):
                samples.extend(local_data)
            if REF_LIST in local_data:
                for data_b in local_data[REF_LIST]:
                    samples.extend(data_b)
        if self._rollout_batch_size < 0:
            self._rollout_batch_size = len(samples)
        self._buffer += samples
        return samples

    def queue_not_empty(self):
        return self.queue.qsize() > 0

    def shuffle(self):
        random.shuffle(self._buffer)

    def get_samples(self, start_index, end_index):
        return self._buffer[start_index: end_index]

    def __len__(self):
        return len(self._buffer)

    def sync(self):
        while self.queue_not_empty():
            self.add_raw_batch()

    @property
    def buffer(self):
        return self._buffer

    @property
    def episode_id(self):
        return self._episode_id


class ReplaySampleManager:
    """
    replay sample Manager, users should inherit it to self-defined replay samples for trainer
    """
    def __init__(self, global_args):
        self.args = global_args
        self._metric_prefix = "replay"
        self._metric_list = []

    def __call__(self, episode_replay_buffers: List[EpisodeReplayBuffer]) -> List[Dict]:
        raise NotImplementedError("default replay sample function is not currently supported")

    def get_and_clear_metrics(self):
        if self._metric_list is None or len(self._metric_list) == 0:
            return self._metric_prefix, {}

        reduced_metrics = map_reduce_metrics(self._metric_list)
        self._metric_list = []
        return self._metric_prefix, reduced_metrics


class RLHFDataLoader:
    """
    RLHF data loader
    """

    def __init__(
        self,
        datasets,
        sampler,
        collate_fn=None,
        data_parallel_rank: int=0,
        data_parallel_size: int=1,
        num_inference_per_prompt: int=1,
    ):
        """generate prompts data loader"""

        self.datasets = datasets
        self.sampler = sampler
        self.collate_fn = collate_fn if collate_fn is not None else default_collate
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.uid = 0
        self.num_inference_per_prompt = num_inference_per_prompt

    def __iter__(self):
        self.sampler_iter = iter(self.sampler)
        while True:
            try:
                batch_idxes = next(self.sampler_iter)
                batch = []
                for dataset_idx, data_idx, _ in batch_idxes:
                    data = copy.deepcopy(self.datasets[dataset_idx][data_idx])
                    data['uid'] = self.uid * self.data_parallel_size + self.data_parallel_rank
                    self.uid += 1
                    batch.append(data)
                yield self.collate_fn(batch)
            except StopIteration:
                self.sampler_iter = iter(self.sampler)
