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
"""module runtime decorator"""

import traceback

import torch
from torch.cuda import nvtx
import ray

from chatlearn.utils import future
from chatlearn.utils import utils
from chatlearn.utils.global_vars import _EXIT_ACTOR_NAME
from chatlearn.utils.utils import execute


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
                execute("ray stop")
                raise

    return inner


def timeit(func, func_name):
    def inner(self, *args, **kwargs):
        if self.runtime_args.nsys:
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
        if self.runtime_args.nsys:
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


def preprocess_compute(func, is_forward_step, trainable):
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

        def get_kwarg(key):
            return kwargs.pop(key) if key in kwargs else False
        to_empty_cache = get_kwarg('to_empty_cache')
        to_onload = get_kwarg('to_onload')
        to_offload = get_kwarg('to_offload')
        is_last_batch = get_kwarg('is_last_batch')
        is_eval = get_kwarg('is_eval')

        if to_onload:
            self.onload()
        generation_batch_size = self.module_args.generation_batch_size
        final_results = None
        if not trainable and generation_batch_size:
            # split into micro-batches if generation_batch_size < input_batch, then concat the results
            # this happens when different models have difference batch sizes
            input_batch = 0
            for value in args[0].values():
                input_batch = len(value)
                break
            input_data = args[0]
            if input_batch > generation_batch_size and not hasattr(self, 'generate_vllm'):
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
                # for model with DP, we need to return results from all ranks
                # for model with TP/PP, only return the results from last rank
                if self.is_last_rank() or self.data_parallel_size is None or self.data_parallel_size > 1:
                    final_results = concat_along_batch(results)
            else:
                if is_forward_step:
                    kwargs["iteration"] = self._iteration
                ret = func(self, *args, **kwargs)
                ret = utils.to_device('cpu', ret)
                self._iteration += 1
                final_results = None
                # for model with DP, we need to return results from all ranks
                # for model with TP/PP, only return the results from last rank
                if self.is_last_rank() or self.data_parallel_size is None or self.data_parallel_size > 1:
                    final_results = ret
        else:
            kwargs["iteration"] = self._train_iteration
            self._train_iteration += 1
            ret = func(self, *args, **kwargs)
            ret = utils.to_device('cpu', ret)
            if self.is_last_rank():
                final_results = ret
        if to_empty_cache:
            self.empty_cache()
        if to_offload:
            self.offload()
        if is_last_batch and not is_eval:
            self.runtime_args.consumed_samples += self.runtime_args.sample_per_episode
        return final_results

    return inner


def decorate_class_func(cls, func_name, decorator, *args, **kwargs):
    if not hasattr(cls, func_name):
        return
    func = getattr(cls, func_name)
    if func.__qualname__.startswith(decorator.__name__):
        # already decorated
        # This usually occurs when one class inherits from another class,
        # for example, if 'reference' inherits from 'policy', then methods like 'offload_optimizer_states'
        # would be decorated in the base class, eliminating the need for repeated decoration.
        return
    setattr(cls, func_name, decorator(func, *args, **kwargs))
