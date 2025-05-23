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
"""Hooks of vllm-0.7.4 cumem sleep and wake_up for param_sync or generation."""


# pylint: disable=wildcard-import,ungrouped-imports
from typing import Optional, Tuple, Union
import torch

from vllm.device_allocator import cumem
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.device_allocator.cumem import create_and_map, libcudart, unmap_and_release
from vllm.utils import is_pin_memory_available

def sleep(
        self,
        offload_tags: Optional[Union[Tuple[str, ...],
                                        str]] = None,
        is_param_sync=False) -> None:
    """
    Put the allocator in sleep mode.
    All data in the memory allocation with the specified tag will be 
    offloaded to CPU memory, and others will be discarded.

    :param offload_tags: The tags of the memory allocation that will be
        offloaded. The rest of the memory allocation will be discarded.
    """
    if offload_tags is None:
        # by default, allocated tensors are offloaded
        # when the allocator sleeps
        offload_tags = (CuMemAllocator.default_tag, )
    elif isinstance(offload_tags, str):
        offload_tags = (offload_tags, )

    assert isinstance(offload_tags, tuple)

    for ptr, data in self.pointer_to_data.items():
        if is_param_sync and data.tag not in ["weights"]:
            continue
        handle = data.handle
        if data.tag in offload_tags:
            size_in_bytes = handle[1]
            cpu_backup_tensor = torch.empty(
                size_in_bytes,
                dtype=torch.uint8,
                device='cpu',
                pin_memory=is_pin_memory_available())
            cpu_ptr = cpu_backup_tensor.data_ptr()
            libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
            data.cpu_backup_tensor = cpu_backup_tensor
        unmap_and_release(handle)

cumem.CuMemAllocator.sleep = sleep

def _wake_up(self, tags: Optional[list[str]] = None, is_param_sync=False):
    """
    Wake up the allocator from sleep mode.
    All data that is previously offloaded will be loaded back to GPU 
    memory, and the rest of the data will have empty memory."""
    wake_up_tags = None
    if is_param_sync:
        wake_up_tags = ["weights"]
    for ptr, data in self.pointer_to_data.items():
        if wake_up_tags is not None and data.tag not in wake_up_tags:
            continue
        handle = data.handle
        create_and_map(handle)
        if data.cpu_backup_tensor is not None:
            # assert data.tag in wake_up_tags, f"while {data.tag}"
            cpu_backup_tensor = data.cpu_backup_tensor
            if cpu_backup_tensor is not None:
                size_in_bytes = cpu_backup_tensor.numel(
                ) * cpu_backup_tensor.element_size()
                cpu_ptr = cpu_backup_tensor.data_ptr()
                libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                data.cpu_backup_tensor = None

cumem.CuMemAllocator.wake_up = _wake_up
