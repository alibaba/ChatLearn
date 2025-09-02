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
"""Torch module"""

import gc
import os
from typing import Optional

import ray
import torch
import torch.distributed as dist

from chatlearn.utils.logger import log_rank_0, debug_rank_0
from chatlearn.utils.utils import get_full_proc_memory_info
from chatlearn.runtime.decorator import timeit
from .base_module import BaseModule

class TorchModule(BaseModule):
    """TorchModule is the class for Alignment Torch models.

    Args
    ----
    name : str
        model name
    """
    # pylint: disable=abstract-method

    def model_setup(self):
        """
        :meta private:
        """
        super().model_setup()
        if self.runtime_args.profiler_dir is not None and self.replica_id == 0:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=1,
                    repeat=1),
                    profile_memory=False,
                    record_shapes=False,
                    with_stack=False,
                    with_flops=False,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.runtime_args.profiler_dir)
            )
            self.profiler.start()

    def get_visible_gpus(self):
        """
        :meta private:
        """
        return ray.get_gpu_ids()

    def set_env(self, args):
        """
        :meta private:
        """
        for key, value in args.items():
            os.environ[key] = str(value)
        return True

    def get_dist_env(self):
        """
        :meta private:
        """
        envs = {}
        for key in ['RANK', 'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'LOCAL_RANK']:
            envs[key] = os.environ[key]
        return envs

    def peak_memory(self):
        """
        :meta private:
        """
        self._peak_memory = max(self._peak_memory, torch.cuda.max_memory_allocated() / (1024 ** 3))
        return self._peak_memory

    def empty_cache(self):
        """
        :meta private:
        """
        if not self.timers("empty_cache").started_:
            self.timers("empty_cache").start()
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        debug_rank_0(f"{self.name} replica: {self.replica_id}, before empty cache, peak mem: {peak_mem:.2f} GiB",
                   self._logger)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        debug_rank_0(f"{self.name} replica: {self.replica_id}, after empty cache, peak mem: {peak_mem:.2f} GiB",
                   self._logger)
        self.timers("empty_cache").stop()

    @property
    def world_size(self):
        return dist.get_world_size()

    def get_rank(self):
        return dist.get_rank()

    def _get_if_not_none(self, to_set: Optional[bool], default: bool) -> bool:
        if not default:
            return False
        if to_set is not None:
            return to_set
        return default

    @timeit()
    def onload(self,
               to_onload_weights: Optional[bool] = None,
               to_build_grad_buffers: Optional[bool] = None,
               to_onload_main_weights: Optional[bool] = None,
               to_onload_optimizer_states: Optional[bool] = None):

        if not self.is_colocate:
            return
        to_onload_weights = self._get_if_not_none(to_onload_weights, self.module_args.free_gpu_memory.offload_weights)
        to_build_grad_buffers = self._get_if_not_none(to_build_grad_buffers, self.module_args.free_gpu_memory.free_grad_buffers)
        to_onload_main_weights = self._get_if_not_none(to_onload_main_weights, self.module_args.free_gpu_memory.offload_weights)
        to_onload_optimizer_states = self._get_if_not_none(to_onload_optimizer_states, self.module_args.free_gpu_memory.offload_optimizer_states)
        if to_onload_weights or to_build_grad_buffers or to_onload_main_weights or to_onload_optimizer_states:
            log_rank_0(get_full_proc_memory_info('Before onload'), self._logger)
            torch.cuda.synchronize()
            timer = self.timers(f'{self.name}_free_memory')
            if not timer.started_:
                timer.start()
            torch.distributed.barrier()
            if to_onload_weights:
                self.onload_weights()
            if self.trainable:
                if to_build_grad_buffers:
                    self.build_grad_buffers()
                if to_onload_main_weights:
                    self.onload_main_weights()
                if to_onload_optimizer_states:
                    self.onload_optimizer_states()
            torch.distributed.barrier()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            timer.stop()
            log_rank_0(get_full_proc_memory_info('After onload'), self._logger)

    @timeit()
    def offload(self,
               to_offload_weights: Optional[bool] = None,
               to_free_grad_buffers: Optional[bool] = None,
               to_offload_main_weights: Optional[bool] = None,
               to_offload_optimizer_states: Optional[bool] = None):
        # The first time of calling `offload_weights` and `offload_main_weights` has a higher peak memory.
        # So `free_grad_buffers` is called first to free memory, and `offload_weights` is called afterward
        # to make more space for `offload_main_weights`.
        if not self.is_colocate:
            return
        to_offload_weights = self._get_if_not_none(to_offload_weights, self.module_args.free_gpu_memory.offload_weights)
        to_offload_main_weights = self._get_if_not_none(to_offload_main_weights, self.module_args.free_gpu_memory.offload_weights)
        to_free_grad_buffers = self._get_if_not_none(to_free_grad_buffers, self.module_args.free_gpu_memory.free_grad_buffers)
        to_offload_optimizer_states = self._get_if_not_none(to_offload_optimizer_states, self.module_args.free_gpu_memory.offload_optimizer_states)
        if to_free_grad_buffers or to_offload_weights or to_offload_optimizer_states or to_offload_main_weights:
            log_rank_0(get_full_proc_memory_info('Before offload'), self._logger)
            torch.cuda.synchronize()
            timer = self.timers(f'{self.name}_free_memory')
            if not timer.started_:
                timer.start()
            torch.distributed.barrier()
            if self.trainable:
                if to_free_grad_buffers:
                    self.free_grad_buffers()
                if to_offload_main_weights:
                    self.offload_main_weights()
                if to_offload_optimizer_states:
                    self.offload_optimizer_states()
            if to_offload_weights:
                self.offload_weights()
            torch.distributed.barrier()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            timer.stop()
            log_rank_0(get_full_proc_memory_info('After offload'), self._logger)
