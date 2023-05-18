import math
import random
from itertools import cycle

import ray
import torch
from torch.nn.utils.rnn import pad_sequence

from rlhf.utils import future


def get_iter_keys(data):
  if isinstance(data, (list, tuple)):
    return range(len(data))
  elif isinstance(data, dict):
    return data.keys()


def create_from_type(data):
  if isinstance(data, (list, tuple)):
    return [None] * len(data)
  return type(data)()


def batching(tensors, padding_value=0.0, padding_type="right"):
  if isinstance(tensors[0], torch.Tensor):
      if tensors[0].dim() == 0:
          return torch.stack(tensors)
      if padding_type == "right":
          return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
      else:
          return pad_sequence([elem.flip(0) for elem in tensors],
                              padding_value=padding_value,
                              batch_first=True).flip(1)
  else:
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
  assert isinstance(batch, (list, tuple, dict)), \
    "batch type {} is not supported".format(type(batch))
  samples = []
  if isinstance(batch, (list, tuple)):
      bs = len(batch[0])
      keys = range(len(batch))
  else:
      bs = len(next(iter(batch.values())))
      keys = batch.keys()

  for b in range(bs):
      if isinstance(batch, (list, tuple)):
          sample = [batch[key][b] for key in keys]
      else:
          sample = {key: batch[key][b] for key in keys}
      samples.append(sample)

  return samples


@ray.remote
class StreamDataset():
    """dataset built from queues"""

    def __init__(self, queue, total_samples, batch_size, padding_config={}, cache=False):
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
        self._padding_config = padding_config
        self._padding_value = {key:value["padding_value"] for key, value in self._padding_config.items()}
        self._padding_type = {key:value["padding_type"] for key, value in self._padding_config.items()}
        self._has_next = True


    def shuffle(self):
        random.shuffle(self.relay_buffer)
        self.iter = self.__iter__()
        self._has_next = True


    def __iter__(self):
        if self._dynamic_dataset:
            return self.iter_dynamic()
        else:
            return self.iter_fixed()


    def iter_fixed(self):
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
                        raise Exception("WARN: data queue is empty")
                    # get from queue
                    data = self.queue.get()
                    merged_data = {}
                    for d in data:
                        local_data = future.get(d)
                        merged_data.update(local_data)
                    samples = split_batch(merged_data)
                    self.relay_buffer += samples
            start_index = self.produce_index
            end_index = min(self.produce_index + self.batch_size, len(self.relay_buffer))
            data_to_batch = self.relay_buffer[start_index: end_index]
            if len(data_to_batch) < self.batch_size:
                data_to_batch += self.relay_buffer[:self.batch_size-len(data_to_batch)]
            batched_data = batching(data_to_batch, self._padding_value, self._padding_type)
            yield batched_data
            batch_count += 1
            self.produce_index += len(data_to_batch)
        assert batch_count == self._num_batches
        assert self.produce_index == len(self.relay_buffer)


    def iter_dynamic(self):
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
                for d in data:
                    local_data = future.get(d)
                    merged_data.update(local_data)
                samples = split_batch(merged_data)
                self.relay_buffer += samples
            start_index = self.produce_index
            end_index = min(self.produce_index + self.batch_size, len(self.relay_buffer))
            data_to_batch = self.relay_buffer[start_index: end_index]
            if len(data_to_batch) < self.batch_size:
                data_to_batch += self.relay_buffer[:self.batch_size-len(data_to_batch)]
            batched_data = batching(data_to_batch, self._padding_value, self._padding_type)
            yield batched_data
            batch_count += 1
            self.produce_index += len(data_to_batch)
        self.total_samples = len(self.relay_buffer)
        self.num_batches = batch_count


    def next(self):
        try:
            data = next(self.iter)
            return data
        except StopIteration:
            self._has_next = False
            return None

    def has_next(self):
        return self._has_next



class RLHFDataLoader:

    def __init__(self, dataset, batch_size):
        """generate prompts data loader"""
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_iter = cycle(iter(self.dataset))


    def __iter__(self):
        batch_data = []
        for i, item in enumerate(self.data_iter):
            batch_data.append(item)
            if len(batch_data) == self.batch_size:
                batched = batching(batch_data)
                yield batched
                batch_data = []
        if len(batch_data) > 0:
            batched = batching(batch_data)
            yield batched

