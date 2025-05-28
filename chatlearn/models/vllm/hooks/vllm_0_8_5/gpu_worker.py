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
"""Hooks of vllm-0.8.5 gpu_worker sleep and wake_up for param_sync or generation."""

import torch

# pylint: disable=unused-import,wildcard-import,unused-argument
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.utils import GiB_bytes
from vllm.v1.worker import gpu_worker
from chatlearn.utils.logger import logger

def sleep(self, level: int = 1, is_param_sync=False) -> None:
    free_bytes_before_sleep = torch.cuda.mem_get_info()[0]
    allocator = CuMemAllocator.get_instance()
    allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple(), is_param_sync=is_param_sync)
    free_bytes_after_sleep, total = torch.cuda.mem_get_info()
    freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
    used_bytes = total - free_bytes_after_sleep
    assert freed_bytes >= 0, "Memory usage increased after sleeping."
    logger.info(
        f"Sleep mode freed {freed_bytes / GiB_bytes:.2f} GiB memory, \
            {used_bytes / GiB_bytes:.2f} GiB memory is still in use, is_param_sync: {is_param_sync}."
    )

gpu_worker.Worker.sleep = sleep

def wake_up(self, is_param_sync=False) -> None:
    free_bytes_before_wakeup = torch.cuda.mem_get_info()[0]
    allocator = CuMemAllocator.get_instance()
    allocator.wake_up(is_param_sync=is_param_sync)
    free_bytes_after_wakeup = torch.cuda.mem_get_info()[0]
    freed_bytes = free_bytes_before_wakeup - free_bytes_after_wakeup
    logger.info(
        "Sleep mode wakeup cost %.2f GiB memory, ", freed_bytes / GiB_bytes)

gpu_worker.Worker.wake_up = wake_up
