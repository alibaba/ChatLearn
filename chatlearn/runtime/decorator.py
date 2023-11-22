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
"""module runtime decorator"""

import traceback

import torch
from torch.cuda import nvtx
import ray

from chatlearn.utils import future
from chatlearn.utils import utils
from chatlearn.utils.global_vars import _EXIT_ACTOR_NAME
from chatlearn.utils.logger import logger


def monitor_error(func, func_name):
    def inner(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self._logger.exception(f"Catch exception ========= in {self.name} {func_name} {e}")
            exit_actor = ray.get_actor(_EXIT_ACTOR_NAME)
            traceback_msg =  f"{traceback.format_exc()}"
            address = self.get_address()
            ray.get(exit_actor.add_error_node_and_msg.remote(address, traceback_msg))
            future.wait(self.error_signal.set_address.remote(address))
            # for other error, we raise in the corresponding workers
            if self.is_master_node():
                for line in traceback_msg.split("\n"):
                    self._logger.exception(line)
                raise

    return inner


def timeit(func, func_name):
    def inner(self, *args, **kwargs):
        if self.rlhf_args.nsys:
            nvtx.range_push(func_name)
        if self.is_last_rank():
            # for the class inherited from base, it may call multiple times, so use the first start time
            if not self.timers(func_name).started_:
                self.timers(func_name).start()
            ret = func(self, *args, **kwargs)
            self.timers(func_name).stop()
        else:
            ret = func(self, *args, **kwargs)
        if self.profiler is not None and self._iteration > 0 and self._iteration <=2 and self.replica_id == 0 \
            and func_name in ["forward_step", "train_step"]:
            self.profiler.step()
        if self.profiler is not None and self._iteration ==3 and self.replica_id == 0 and func_name in ["forward_step", "train_step"]:
            self.profiler.stop()
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

    accum_bs = 0
    new_batches = []
    while accum_bs < bs:
        if isinstance(batch, (list, tuple)):
            new_batch = [batch[key][accum_bs:min(accum_bs + new_batch_size, bs)] for key in keys]
        else:
            new_batch = {key: batch[key][accum_bs:min(accum_bs + new_batch_size, bs)] for key in keys}
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


def preprocess_compute(func, is_forward_step):
    """
    1. if is_forward_step is True, merge a list of dict into one dict, i.e., merge inputs of forward_step.
    2. split a list of data for data_parallel, this is used for train_step
    3. convert output to cpu
    """

    def inner(self, *args, **kwargs):
        args = future.get(args)
        if is_forward_step and len(args) > 1:
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
        to_empty_cache = kwargs.pop('to_empty_cache') if 'to_empty_cache' in kwargs else False
        is_last_batch = kwargs.pop('is_last_batch') if 'is_last_batch' in kwargs else False
        is_eval = kwargs.pop('is_eval') if 'is_eval' in kwargs else False
        generation_batch_size = self.module_args.generation_batch_size
        final_results = None
        if not self.trainable and generation_batch_size:
            # split into micro-batches if generation_batch_size < input_batch, then concat the results
            # this happens when different models have difference batch sizes
            input_batch = 0
            for value in args[0].values():
                input_batch = len(value)
                break
            input_data = args[0]
            if input_batch > generation_batch_size:
                args = list(args)
                batches = split_along_batch(input_data, generation_batch_size)
                results = []
                for batch in batches:
                    args[0] = batch
                    if is_forward_step:
                        kwargs["iteration"] = self._iteration
                    ret = func(self, *args, **kwargs)
                    self._iteration += 1
                    ret = utils.to_device('cpu', ret)
                    results.append(ret)
                if self.is_last_rank():
                    final_results = concat_along_batch(results)
            else:
                if is_forward_step:
                    kwargs["iteration"] = self._iteration
                ret = func(self, *args, **kwargs)
                ret = utils.to_device('cpu', ret)
                self._iteration += 1
                if self.is_last_rank():
                    final_results = ret
        else:
            ret = func(self, *args, **kwargs)
            ret = utils.to_device('cpu', ret)
            if self.is_last_rank():
                final_results = ret
        if to_empty_cache:
            self.empty_cache()
        if is_last_batch and not is_eval:
            self.rlhf_args.consumed_samples += self.rlhf_args.sample_per_episode
        return final_results

    return inner


def decorate_class_func(cls, func_name, decorator, *args, **kwargs):
    func = getattr(cls, func_name)
    if func.__qualname__.startswith(decorator.__name__):
        # already decorated
        logger.warning(f"{func_name} {func} already decorated with {decorator}")
        return
    setattr(cls, func_name, decorator(func, *args, **kwargs))
