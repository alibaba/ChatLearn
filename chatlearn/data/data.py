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
from typing import List, Dict

import ray
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate
from chatlearn.utils import future
from chatlearn.utils.constant import CHATLEARN_REGROUP_TAG
from chatlearn.utils.utils import regroup_by_concat_along_batch, map_reduce_metrics


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


def batch_shuffle(data, batch_size):
    num_batches = len(data) // batch_size
    batches = [data[batch_size*i:batch_size*(i+1)] for i in range(num_batches)]
    random.shuffle(batches)

    shuffled_data = [item for batch in batches for item in batch]

    return shuffled_data


@ray.remote
class StreamDataset:
    """dataset built from queues"""

    def __init__(self, data_loader_type, micro_batch_size, padding_config=None, max_relay_episode=0, relay_episode_offset=0, global_batch_size=-1):
        """
        Args:
            data_loader_type: fixed or dynamic
        """
        if data_loader_type == "fixed":
            self._dynamic_dataset = False
        else:
            self._dynamic_dataset = True
        self.batch_size = micro_batch_size

        self._padding_config = padding_config if padding_config is not None else {}
        self._padding_value = {key: value["padding_value"] for key, value in padding_config.items()}
        self._padding_type = {key: value["padding_type"] for key, value in padding_config.items()}

        if max_relay_episode < 0:
            max_relay_episode = math.inf
        self._max_relay_episode = max_relay_episode
        self._relay_episode_offset = relay_episode_offset
        self._episode_relay_buffers = []
        self.relay_sample_manager = None

        # ChunkFlow: Params for ChunkFlow
        self.prefetch_batch_cnt= micro_batch_size if global_batch_size < 0 else global_batch_size

    def shuffle(self, batch_size=None):
        """
        shuffle relay buffer
        """
        self.relay_buffer.shuffle(batch_size)
        self.iter = self.__iter__() # pylint: disable=unnecessary-dunder-call
        self._has_next = True

    def __iter__(self):
        if self._dynamic_dataset and not self._read_data_complete:
            return self.iter_dynamic()
        return self.iter_fixed()

    def _get_batch(self, start_index):
        end_index = min(start_index + self.batch_size, len(self.relay_buffer))
        data_to_batch = self.relay_buffer.get_samples(start_index, end_index)
        if len(data_to_batch) < self.batch_size:
            data_to_batch += self.relay_buffer.get_samples(0, self.batch_size - len(data_to_batch))
        batched_data = batching(data_to_batch, self._padding_value, self._padding_type)
        return batched_data

    def iter_fixed(self):
        """
        iteration with fixed batch size
        """
        produce_index = 0
        batch_count = 0
        prefetched_batch_list = []
        while produce_index < self._total_samples:
            # read from cache
            if len(self.relay_buffer) < self._total_samples:
                while len(self.relay_buffer) < self._total_samples and \
                    (len(self.relay_buffer) - produce_index) < self.batch_size:
                    self.relay_buffer.add_raw_batch()
            prefetched_batch_list.append(self._get_batch(produce_index))
            if len(prefetched_batch_list) == self.prefetch_batch_cnt:
                # ChunkFlow: Sort by sample length for better balance across data parallel ranks
                # TODO: fix hardcode key for sample len
                if "response_ids" in prefetched_batch_list[0].keys():
                    prefetched_batch_list.sort(key=lambda x: len(x["response_ids"][0]))
                for batched_data in prefetched_batch_list:
                    yield batched_data
                    batch_count += 1
                prefetched_batch_list.clear()
            produce_index += self.batch_size
        if len(prefetched_batch_list) != 0:
            for batched_data in prefetched_batch_list:
                yield batched_data
                batch_count += 1
        assert batch_count == math.ceil(self._total_samples / self.batch_size)
        assert produce_index >= len(self.relay_buffer), \
               f"produce_index: {produce_index} < len(self.relay_buffer) {len(self.relay_buffer)}"

    def iter_dynamic(self):
        """
        iteration with dynamic batch size
        """
        produce_index = 0
        if self._read_data_complete:
            return self.iter_fixed()
        batch_count = 0

        while self.relay_buffer.queue_not_empty():
            while self.relay_buffer.queue_not_empty() and \
                (len(self.relay_buffer) - produce_index) < self.batch_size:
                # get from queue
                self.relay_buffer.add_raw_batch()
            batched_data = self._get_batch(produce_index)
            yield batched_data
            batch_count += 1
            produce_index += self.batch_size
        self._read_data_complete = True
        assert len(self.relay_buffer) == self._total_samples
        self._num_batches = batch_count

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

    def set_dataset(self, queue, episode_id, relay_sample_manager=None, sample_per_episode=-1):
        relay_buffer = EpisodeRelayBuffer(episode_id, queue=queue)
        if self._max_relay_episode > 0 and episode_id >= self._relay_episode_offset:
            self._episode_relay_buffers.append(relay_buffer)
            if len(self._episode_relay_buffers) > self._max_relay_episode:
                old_buffer = self._episode_relay_buffers.pop(0)
                del old_buffer

            # this function will sync until all data computing finished,
            # which will block training until environment rollout finished.
            if os.getenv("SKIP_GENERATION", None) is None:
                relay_buffer.sync()
            if relay_sample_manager is None:
                raise Exception("default relay sample function is not currently supported")

            self.relay_sample_manager = relay_sample_manager
            buffer = self.relay_sample_manager(self._episode_relay_buffers)
            self.relay_buffer = EpisodeRelayBuffer(episode_id, buffer=buffer)
            self._total_samples = len(self.relay_buffer)
            self._read_data_complete = True
        else:
            num_rollout_batches = queue.qsize()
            self.relay_buffer = relay_buffer
            self.relay_buffer.add_raw_batch()
            assert sample_per_episode != -1, "In fixed batch size, you must set sample_per_episode for StreamDataset."
            self._total_samples = sample_per_episode
            self._read_data_complete = num_rollout_batches <= 1
        self.iter = iter(self)
        self._has_next = True

    def episode_relay_buffers(self):
        return self._episode_relay_buffers

    def total_samples(self):
        return self._total_samples

    def batch_per_episode(self):
        return math.ceil(self._total_samples / self.batch_size)

    def get_and_clear_metrics(self):
        # TODO: deal with situation that relay_sample_manager is None
        try:
            return self.relay_sample_manager.get_and_clear_metrics()
        except Exception:
            return "no relay", {}

class EpisodeRelayBuffer:
    """EpisodeRelayBuffer"""

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
        merged_data = {}
        for item in data:
            local_data = future.get(item)
            if CHATLEARN_REGROUP_TAG in local_data:
                local_data = regroup_by_concat_along_batch(local_data[CHATLEARN_REGROUP_TAG])
            merged_data.update(local_data)
        samples = split_batch(merged_data)
        if self._rollout_batch_size < 0:
            self._rollout_batch_size = len(samples)
        self._buffer += samples
        return samples

    def queue_not_empty(self):
        return self.queue.qsize() > 0

    def shuffle(self, batch_size):
        if batch_size is None:
            random.shuffle(self._buffer)
            return
        self._buffer = batch_shuffle(self._buffer, batch_size)

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


class RelaySampleManager:
    """
    Relay sample Manager, users should inherit it to self-defined relay samples for trainer
    """
    def __init__(self, global_args):
        self.args = global_args
        self._metric_prefix = "relay"
        self._metric_list = []

    def __call__(self, episode_relay_buffers: List[EpisodeRelayBuffer]) -> List[Dict]:
        raise NotImplementedError("default relay sample function is not currently supported")

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
        add_uid=False,
        data_parallel_rank=0,
        data_parallel_size=1,
        num_inference_per_prompt=1,
        vllm_prompt_key="prompt"):
        """generate prompts data loader"""

        self.datasets = datasets
        self.dataset_num = len(self.datasets)
        self.sampler = sampler
        self.collate_fn = collate_fn
        self.add_uid = add_uid
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.num_inference_per_prompt = num_inference_per_prompt
        self.vllm_prompt_key = vllm_prompt_key

    def __iter__(self):
        self.sampler_iter = iter(self.sampler)
        while True:
            try:
                batch_idxes = next(self.sampler_iter)
                batch = [self.datasets[dataset_idx][data_idx] for dataset_idx, data_idx, _ in batch_idxes]
                id_in_episode = [id for _, _, id in batch_idxes]
                if self.add_uid:
                    batch = self.update_data_uid(batch, id_in_episode)
                if self.collate_fn is not None:
                    yield self.collate_fn(batch)
                else:
                    yield default_collate(batch)
            except StopIteration:
                self.sampler_iter = iter(self.sampler)

    def update_data_uid(self, batch, id_in_episode):
        updated_batch = []
        for i, data in enumerate(batch):
            if isinstance(data, dict) and self.vllm_prompt_key in data \
                and isinstance(data[self.vllm_prompt_key], dict):
                copy_data = copy.deepcopy(data)
                copy_data[self.vllm_prompt_key]['uid'] = str(id_in_episode[i])
                updated_batch.append(copy_data)
            else:
                updated_batch.append(data)
        return updated_batch
