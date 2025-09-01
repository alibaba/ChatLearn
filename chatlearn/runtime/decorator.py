# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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

import inspect
import traceback
import functools

from torch.cuda import nvtx
import ray

from chatlearn.utils import future
from chatlearn.utils import utils
from chatlearn.utils.global_vars import _EXIT_ACTOR_NAME, set_wrap_func
from chatlearn.utils.utils import execute

def monitor_error():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            func_name = func.__name__
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
        return wrapper
    return decorator


def timeit():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            func_name = func.__name__
            if self.runtime_args.nsys:
                nvtx.range_push(func_name)

            if not self.timers(func_name).started_:
                self.timers(func_name).start()
            ret = func(self, *args, **kwargs)
            self.timers(func_name).stop()

            if self.profiler is not None and self._iteration > 0 and self._iteration <=2 and self.replica_id == 0 \
                and func_name in ["forward_step", "train_step"]:
                self.profiler.step()
            if self.profiler is not None and self._iteration ==3 and self.replica_id == 0 and func_name in ["forward_step", "train_step"]:
                self.profiler.stop()
            if self.runtime_args.nsys:
                nvtx.range_pop()
            return ret

        @functools.wraps(func)
        async def wrapper_async(self, *args, **kwargs):
            func_name = func.__name__
            if self.runtime_args.nsys:
                nvtx.range_push(func_name)

                if not self.timers(func_name).started_:
                    self.timers(func_name).start()
                ret = await func(self, *args, **kwargs)
                self.timers(func_name).stop()

            if self.profiler is not None and self._iteration>0 and self._iteration<=2 and self.replica_id==0 \
                and func_name in ["forward_step", "train_step"]:
                self.profiler.step()
            if self.profiler is not None and self._iteration==3 and self.replica_id==0 and func_name in ["forward_step", "train_step"]:
                self.profiler.stop()
            if self.runtime_args.nsys:
                nvtx.range_pop()
            return ret
        wrapper = wrapper_async if inspect.iscoroutinefunction(func) else wrapper
        return wrapper

    return decorator

def compute_decorator(trainable, rollout):
    def decorator(func):
        """
        1. if not trainable, merge a list of dict into one dict, i.e., merge inputs of forward_step.
        2. split a list of data for data_parallel, this is used for train_step
        3. convert output to cpu
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            data_list = self.data_fetch(args, train_func=trainable)
            # Get kwargs for function
            to_empty_cache = kwargs.pop('to_empty_cache', False)
            to_onload = kwargs.pop('to_onload', False)
            to_offload = kwargs.pop('to_offload', False)
            is_last_batch = kwargs.pop('is_last_batch', False)
            is_eval = kwargs.pop('is_eval', False)

            # Onload model if needed
            if to_onload:
                self.onload()

            # Run decorated function
            if 'iteration' in inspect.signature(func).parameters:
                kwargs["iteration"] = self._iteration
            args = [data_list]
            ret = func(self, *args, **kwargs)
            ret = utils.to_device('cpu', ret)
            self._iteration += 1
            # Return result for every rank
            final_results = ret

            # Clean up after function
            # TODO, remove
            if to_empty_cache and not rollout:
                self.empty_cache()

            if to_offload and not (rollout and is_eval):
                self.offload()

            # TODO fix consumed samples
            if is_last_batch and not is_eval:
                self.runtime_args.consumed_samples += self.runtime_args.sample_per_episode
            return final_results

        @functools.wraps(func)
        async def wrapper_async(self, *args, **kwargs):
            """
            only for async rollout
            """
            data_list = self.data_fetch(args, train_func=trainable)

            is_eval = kwargs.pop('is_eval', False)
            # Onload model if needed
            if kwargs.pop('to_onload', False):
                await self.onload()

            # Run decorated function
            if 'iteration' in inspect.signature(func).parameters:
                kwargs["iteration"] = self._iteration
            args = [data_list]
            ret = await func(self, *args, **kwargs)
            ret = utils.to_device('cpu', ret)
            self._iteration += 1

            # Clean up after function
            # TODO, remove
            if kwargs.pop('to_offload', False) and not is_eval:
                await self.offload()
            # TODO fix consumed samples
            if kwargs.pop('is_last_batch', False) and not is_eval:
                self.runtime_args.consumed_samples += self.runtime_args.sample_per_episode
            return ret
        wrapper = wrapper_async if inspect.iscoroutinefunction(func) else wrapper
        return wrapper
    return decorator

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
    new_func = decorator(func, *args, **kwargs)
    set_wrap_func(func, new_func)
    setattr(cls, func_name, new_func)
