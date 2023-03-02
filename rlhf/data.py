import random
import math
import ray
import torch
from collections.abc import Sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, Dataset


def get_iter_keys(data):
  if isinstance(data, Sequence):
    return range(len(data))
  elif isinstance(data, dict):
    return data.keys()


def create_from_type(data):
  if isinstance(data, Sequence):
    return [None] * len(data)
  return type(data)()


def batching(tensors, padding_value=0.0):
  if isinstance(tensors[0], torch.Tensor):
    if tensors[0].dim() == 0:
      return torch.stack(tensors)
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
  else:
    batch = create_from_type(tensors[0])
    batch_size = len(tensors)
    for key in get_iter_keys(tensors[0]):
      pad = padding_value.get(key, 0.0) if isinstance(padding_value,
                                                      dict) else padding_value
      batched = [tensors[j][key] for j in range(batch_size)]
      if isinstance(batched[0], torch.Tensor):
        batched = batching(batched, pad)
      batch[key] = batched
    return batch


def split_batch(batch):
  assert isinstance(batch, (Sequence, dict)), \
    "batch type {} is not supported".format(type(batch))
  samples = []
  if isinstance(batch, Sequence):
    bs = len(batch[0])
    keys = range(len(batch))
  else:
    bs = len(next(iter(batch.values())))
    keys = batch.keys()

  for b in range(bs):
    if isinstance(batch, Sequence):
      sample = [batch[key][b] for key in keys]
    else:
      sample = {key: batch[key][b] for key in keys}
    samples.append(sample)

  return samples


@ray.remote
class StreamDataset():
    """dataset built from queues"""

    def __init__(self, queue, total_samples, batch_size, cache=False):
        self.queue = queue
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(total_samples / batch_size)
        self.produce_index = 0
        self.cache = cache
        self.relay_buffer = []
        self.iter = self.__iter__()


    def shuffle(self):
        random.shuffle(self.relay_buffer)
        self.iter = self.__iter__()


    def __iter__(self):
        self.produce_index = 0
        if len(self.relay_buffer) == self.total_samples:
            self.cache = False
        batch_count = 0
        while self.produce_index < self.total_samples:
            # read from cache
            if len(self.relay_buffer) < self.total_samples:
                while len(self.relay_buffer) < self.total_samples and \
                        (len(self.relay_buffer) - self.produce_index) < self.batch_size:
                    # get from queue
                    data = self.queue.get()
                    merged_data = {}
                    for d in data:
                        local_data = ray.get(d)
                        merged_data.update(local_data)
                    samples = split_batch(merged_data)
                    self.relay_buffer += samples
            start_index = self.produce_index
            end_index = min(self.produce_index + self.batch_size, len(self.relay_buffer))
            data_to_batch = self.relay_buffer[start_index: end_index]
            batched_data = batching(data_to_batch)
            yield batched_data
            batch_count += 1
            self.produce_index += len(data_to_batch)
        assert batch_count == self.num_batches
        assert self.produce_index == len(self.relay_buffer)


    def next(self):
        try:
            data = next(self.iter)
            return data
        except StopIteration:
            return None


class RLHFDataLoader:

    def __init__(self, dataset, batch_size):
        """generate prompts data loader"""
        if isinstance(dataset, Dataset):
            self.dataset = ray.data.from_torch(dataset).repeat()
        self.batch_size = batch_size


    def __iter__(self):
        for batch in self.dataset.iter_batches(batch_size=self.batch_size):
            yield batching(batch)
