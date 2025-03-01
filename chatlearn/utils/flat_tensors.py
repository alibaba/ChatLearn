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
"""
Provides utility classes for managing multiple tensors and their copying between CPU memory and GPU memory.
"""
import sys
from typing import List

import torch
import torch.distributed as dist


def _pin(t: torch.Tensor):
    """
    Pin the memory of tensor in-place.
    See: https://github.com/pytorch/pytorch/issues/32167
    """
    cudart = torch.cuda.cudart()
    r = cudart.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
    assert r == 0, f'pin memory error, error code: {r.value}'

def _unpin(t: torch.Tensor):
    """
    Un-pin the pinned memory.
    """
    assert t.is_pinned()
    cudart = torch.cuda.cudart()
    r = cudart.cudaHostUnregister(t.data_ptr())
    assert r == 0, f'unpin memory error, error code: {r.value}'


class FlatTensors:
    """
    Manage a list of Tensors for situations where offloading and/or sharding is
    performed.

    Two blocks of memory are allocated: GPU buffer and primary store. GPU
    buffer always stores full data corresponding to the data of tensors, which
    is about to be used soon. GPU buffer is allocated and deallocated depending
    on usages. Primary store is allocated at the beginning, and is kept through
    the whole training. Primary store only stores necessary data when sharding
    is enabled, and its location is 'cpu' if offloading is enabled.

    An empty list of tensors is supported.
    """

    _EMPTY_TENSOR = torch.Tensor()

    def __init__(
        self,
        tensors: List[torch.Tensor],
        primary_store_device='cuda',
        primary_store_shard_group=None,
    ):
        """
        Args:
            tensors: the list of tensors to be managed.
            primary_store_device: which device to allocate primary store.
            primary_store_shard_group: the communication group in which the
             primary store is sharded.
        """
        self._tensors = [*tensors]
        self._shapes = [t.shape for t in tensors]
        self._numels = [t.numel() for t in tensors]

        self._comm_group = primary_store_shard_group
        self.total_numel = self._get_total_numel()
        self._comm_range = (
            self._get_shard_range()
            if primary_store_shard_group
            else (0, self.total_numel)
        )

        self._dtype = tensors[0].dtype if len(tensors) > 0 else torch.bfloat16
        self._shard_primary_store = primary_store_shard_group is not None
        self._primary_store_device = primary_store_device
        self._primary_store = self._alloc_primary_store(self._dtype)
        self._gpu_buffer = self._alloc_gpu_buffer(self._dtype)

        # Aggregate tensor data to GPU buffer.
        s = 0
        for t, numel, _ in zip(self._tensors, self._numels, self._shapes):
            self._gpu_buffer[s: s + numel].copy_(t.data.view(-1))
            s += numel
        self._link_tensor_data_to_gpu_buffer()
        self._in_gpu = True

    def _get_total_numel(self):
        """
        Get the total numel considering sharding group if any.
        """
        n = sum(self._numels)
        if self._comm_group is None:
            return n
        group_size = dist.get_world_size(self._comm_group)
        padded = (n + group_size - 1) // group_size * group_size
        return padded

    def _get_shard_range(self, comm_group=None):
        if comm_group is None:
            comm_group = self._comm_group
        assert comm_group is not None
        group_size = dist.get_world_size(comm_group)
        rank = dist.get_rank(comm_group)
        assert self.total_numel % group_size == 0
        shard_len = self.total_numel // group_size
        start = shard_len * rank
        end = start + shard_len
        return start, end

    def _alloc_primary_store(self, dtype, shard_group=None):
        if self._shard_primary_store:
            (start, end) = self._comm_range
            numel = end - start
        elif shard_group is not None:
            (start, end) = self._get_shard_range(shard_group)
            numel = end - start
        else:
            numel = self.total_numel
        primary_store = torch.empty(
            size=(numel,),
            device=self._primary_store_device,
            dtype=dtype,
            pin_memory=False,
        )
        if self._primary_store_device == 'cpu' and numel > 0:
            _pin(primary_store)
        return primary_store

    def _link_tensor_data_to_gpu_buffer(self):
        s = 0
        for t, numel, shape in zip(self._tensors, self._numels, self._shapes):
            t.data = self._gpu_buffer[s: s + numel].view(shape)
            s += numel

    def _alloc_gpu_buffer(self, dtype, set_zero=False):
        # TODO(jiqi): consider reuse of GPU buffer.
        fn = torch.zeros if set_zero else torch.empty
        return fn(
            (self.total_numel,),
            dtype=dtype,
            device='cuda',
        )

    @torch.no_grad()
    def copy_to_primary_store(self, non_blocking=True):
        if not self._in_gpu:
            return
        (start, end) = self._comm_range
        if self._shard_primary_store:
            self._primary_store.copy_(
                self._gpu_buffer[start:end], non_blocking=non_blocking
            )
        else:
            self._primary_store.copy_(
                self._gpu_buffer, non_blocking=non_blocking
            )

        for t in self._tensors:
            t.data.record_stream(torch.cuda.current_stream())
        self._gpu_buffer.record_stream(torch.cuda.current_stream())
        self.release_gpu_buffer()

    @torch.no_grad()
    def copy_to_gpu_buffer(self, copy_shard_group=None, non_blocking=True):
        if self._in_gpu:
            return
        if (
            not self._shard_primary_store
            and self._primary_store_device == 'cuda'
        ):
            self._gpu_buffer = self._primary_store
            self._link_tensor_data_to_gpu_buffer()
            self._in_gpu = True
            return

        self._gpu_buffer = self._alloc_gpu_buffer(self._dtype)
        self._link_tensor_data_to_gpu_buffer()
        if copy_shard_group is not None:
            assert not self._shard_primary_store
            (start, end) = self._get_shard_range(copy_shard_group)
        else:
            (start, end) = self._comm_range
        if self._shard_primary_store:
            self._gpu_buffer[start:end].copy_(
                self._primary_store, non_blocking=non_blocking
            )
        elif copy_shard_group:
            self._gpu_buffer[start:end].copy_(
                self._primary_store[start:end], non_blocking=non_blocking
            )
        else:
            self._gpu_buffer.copy_(
                self._primary_store, non_blocking=non_blocking
            )
        self._in_gpu = True

    @torch.no_grad()
    def release_gpu_buffer(self):
        """
        Release tensors on GPU memory.
        """
        assert self._in_gpu
        for t in self._tensors:
            t.data = self._EMPTY_TENSOR
        self._gpu_buffer = None
        self._in_gpu = False

    def __del__(self):
        # Unpin the pinned memory, for unit tests.
        if self._primary_store_device == 'cpu' and self._primary_store.is_pinned():
            _unpin(self._primary_store)

class BucketizedFlatTensors:
    """
    Manage a list of Tensors for situations where offloading and/or sharding is
    performed.

    This class is similar with `FlatTensors` except that it partitions tensors
    into several buckets to avoid high peak memory in creation.

    Two blocks of memory are allocated: GPU buffer and primary store. GPU
    buffer always stores full data corresponding to the data of tensors, which
    is about to be used soon. GPU buffer is allocated and deallocated depending
    on usages. Primary store is allocated at the beginning, and is kept through
    the whole training. Primary store only stores necessary data when sharding
    is enabled, and its location is 'cpu' if offloading is enabled.

    An empty list of tensors is supported.
    """

    def __init__(
        self,
        tensors: List[torch.Tensor],
        bucket_size_mb: int,
        primary_store_device='cuda',
        primary_store_shard_group=None,
    ):
        size_limit = bucket_size_mb * 1024 * 1024 if bucket_size_mb > 0 else sys.maxsize
        self._flat_tensors = []
        bucket = []
        total_size = 0
        for t in tensors:
            size = t.numel() * t.element_size()
            if total_size + size > size_limit and len(bucket) > 0:
                self._flat_tensors.append(
                    FlatTensors(bucket, primary_store_device, primary_store_shard_group)
                )
                total_size = 0
                bucket.clear()

            total_size += size
            bucket.append(t)

        if len(bucket) > 0:
            self._flat_tensors.append(
                FlatTensors(bucket, primary_store_device, primary_store_shard_group)
            )

    @torch.no_grad()
    def copy_to_primary_store(self, non_blocking=True):
        for flat_tensor in self._flat_tensors:
            flat_tensor.copy_to_primary_store(non_blocking=non_blocking)

    @torch.no_grad()
    def copy_to_gpu_buffer(self, copy_shard_group=None, non_blocking=True):
        for flat_tensor in self._flat_tensors:
            flat_tensor.copy_to_gpu_buffer(
                copy_shard_group=copy_shard_group, non_blocking=non_blocking
            )

    @torch.no_grad()
    def release_gpu_buffer(self):
        for flat_tensor in self._flat_tensors:
            flat_tensor.release_gpu_buffer()
