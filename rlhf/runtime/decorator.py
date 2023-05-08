import traceback

import ray
import torch
import torch.cuda.nvtx as nvtx

from rlhf.utils import utils
from rlhf.utils.logger import logger


def monitor_error(func, func_name):

    def inner(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"catch exception ========= in {self.name} {e}, {traceback.format_exc()}")
            ray.get(self.error_signal.set.remote(traceback.format_exc()))
            raise
    return inner


def timeit(func, func_name):

    def inner(self, *args, **kwargs):
        if self.rlhf_args.nsys:
            nvtx.range_push(func_name)
        if self.rank == 0:
            # for the class inherited from base, it may call multiple times, so use the first start time
            if not self.timers(func_name).started_:
                self.timers(func_name).start()
            ret = func(self, *args, **kwargs)
            self.timers(func_name).stop()
        else:
            ret = func(self, *args, **kwargs)
        if self.rlhf_args.nsys:
            nvtx.range_pop()
        return ret

    return inner


def split_along_batch(batch, new_batch_size):
    assert isinstance(batch, (list, tuple, dict)), \
        "batch type {} is not supported".format(type(batch))
    if isinstance(batch, (list, tuple)):
        bs = len(batch[0])
        keys = range(len(batch))
    else:
        bs = len(next(iter(batch.values())))
        keys = batch.keys()
    assert bs % new_batch_size == 0, f"{bs} vs {new_batch_size}"
  
    accum_bs = 0
    new_batches = []
    while accum_bs < bs:
        if isinstance(batch, (list, tuple)):
            new_batch = [batch[key][accum_bs:accum_bs+new_batch_size] for key in keys]
        else:
            new_batch = {key: batch[key][accum_bs:accum_bs+new_batch_size] for key in keys}
        accum_bs += new_batch_size
        new_batches.append(new_batch)
    return new_batches


def concat_along_batch(tensors):
    batched = {}
    for key in tensors[0].keys():
        to_batch = [results[key] for results in tensors]
        if isinstance(to_batch[0], torch.Tensor):
            batched[key] = torch.concat(to_batch)
        elif isinstance(to_batch[0], list):
            batched[key] = []
            for seq in to_batch:
                batched[key].extend(seq)
        else:
            raise Exception(f"unkown types {type(to_batch[0])}to concat")
    return batched


def preprocess_compute(func, merge_input):
    """
    1. if merge_input is True, merge a list of dict into one dict, i.e., merge inputs of forward_step.
    2. split a list of data for data_parallel, this is used for train_step
    3. convert output to cpu
    """
    def inner(self, *args, **kwargs):
        args = utils.get(args)
        if merge_input and len(args) > 1:
            if all(isinstance(arg, dict) for arg in args):
                merged = {}
                for arg in args:
                    merged.update(arg)
                args = [merged]
        if self.data_parallel_size is not None and \
                self.data_parallel_rank is not None and \
                self.data_parallel_size > 1:
            data_list = args[0]
            assert isinstance(data_list, list)
            start_idx, end_idx = utils.split_index(len(data_list), self.data_parallel_size)[self.data_parallel_rank]
            args = list(args)
            sub_data_list = data_list[start_idx: end_idx]
            args[0] = sub_data_list
        generation_batch_size = self.module_args.generation_batch_size
        if not self.trainable and generation_batch_size:
            # split into micro-batches if generation_batch_size < input_batch, then concat the results
            # this happens when different models have difference batch sizes
            input_batch = 0
            for key, value in args[0].items():
                input_batch = len(value)
                break
            input_data = args[0]
            if input_batch > generation_batch_size:
                args = list(args)
                batches = split_along_batch(input_data, generation_batch_size)
                results = []
                for i, batch in enumerate(batches):
                    args[0] = batch
                    ret = func(self, *args, **kwargs)
                    ret = utils.to_device('cpu', ret)
                    results.append(ret)
                new_batch = concat_along_batch(results)
                return new_batch
        ret = func(self, *args, **kwargs)
        ret = utils.to_device('cpu', ret)
        return ret

    return inner


def decorate_class_func(cls, func_name, decorator, *args, **kwargs):
    func = getattr(cls, func_name)
    if func.__qualname__.startswith(decorator.__name__):
        # already decorated
        logger.warn(f"{func_name} {func} already decorated with {decorator}")
        return
    setattr(cls, func_name, decorator(func, *args, **kwargs))
