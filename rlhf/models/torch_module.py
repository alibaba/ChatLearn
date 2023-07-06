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
from rlhf.launcher import dlc_utils
from rlhf.utils.logger import log_rank_0
from rlhf.utils.utils import get_free_port, get_host_addr
from .rlhf_module import RLHFModule

class RLHFTorchModule(RLHFModule):
    """RLHFTorchModule"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_addr_port(self):
        """
        Get node address and port

        :meta private:
        """
        if dlc_utils.in_dlc_env():
            addr = dlc_utils.get_addr()
            port = None
        else:
            addr = get_host_addr()
            port = get_free_port()
        return addr, port

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
        """

    @property
    def data_parallel_rank(self):
        """
        data parallel rank
        """

    def empty_cache(self):
        log_rank_0(f"{self.name} before empty cache, peak mem: {torch.cuda.max_memory_allocated() / (1024 ** 3)}GB")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log_rank_0(f"{self.name} after empty cache, peak mem: {torch.cuda.max_memory_allocated() / (1024 ** 3)}GB")

    def check_param_exists(self, names):
        """
        check if the given names exists in current model
        :meta private
        """
        return all(self.exist_parameter(name) for name in names)
