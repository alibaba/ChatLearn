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
"""RLHF torch module"""

import os
import ray
import torch
from chatlearn.utils.logger import log_rank_0
from .rlhf_module import RLHFModule

class RLHFTorchModule(RLHFModule):
    """RLHFTorchModule is the class for RLHF Torch models.

    Args
    ----
    name : str
        model name
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = None

    def model_setup(self):
        """
        :meta private:
        """
        super().model_setup()
        if self.rlhf_args.profiler_dir is not None and self.replica_id == 0:
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
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.rlhf_args.profiler_dir)
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
        for key in ['RANK', 'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'LOCAL_RANK']:
            assert key in args, f"{key} is not set for RLHFTorchWrapper"
            os.environ[key] = str(args[key])
        self._rank = int(os.environ['RANK'])
        return 1

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

    @property
    def data_parallel_size(self):
        """
        data parallel size

        :meta private:
        """

    @property
    def data_parallel_rank(self):
        """
        data parallel rank

        :meta private:
        """

    def empty_cache(self):
        """
        :meta private:
        """
        if not self.timers("empty_cache").started_:
            self.timers("empty_cache").start()
        log_rank_0(f"{self.name} replica: {self.replica_id}, before empty cache, peak mem: {torch.cuda.max_memory_allocated() / (1024 ** 3)}GB",
                   self._logger)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log_rank_0(f"{self.name} replica: {self.replica_id}, after empty cache, peak mem: {torch.cuda.max_memory_allocated() / (1024 ** 3)}GB",
                   self._logger)
        self.timers("empty_cache").stop()

    def check_param_exists(self, names):
        """
        check if the given names exists in current model
        
        :meta private:
        """
        not_exists = []
        for name in names:
            if not self.exist_parameter(name):
                not_exists.append(name)
        if not_exists:
            log_rank_0(f"parameters not exists: {not_exists} in model {self.name}", self._logger)
            return False
        return True

    def is_last_rank(self):
        """
        Is last rank.
        """
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)
        return True
