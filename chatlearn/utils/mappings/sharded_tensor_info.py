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
"""implementation of ShardedTensorInfo"""
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy

import torch
import numpy as np

@dataclass
class ShardedTensorInfo:
    """Represents a mapping between a local shard and a global tensor,
    modified from `megatron.core.dist_checkpointing.mapping.ShardedTensor`

    Global tensor is assumed to consist of many local tensors distributed
    between different processes, while each sharded_info represents part of
    the local tensor (local shard).

    Args:
        param_id: unique identifier of a global tensor
        dtype: tensor dtype
        local_shape: local shard shape
        global_shape: global tensor shape

        axis_fragmentations: global tensor fragmentation of each axis

        local_offset: offset of this shard in a local tensor
        global_offset: offset of a local tensor in a global tensor,
            specified in number of tensor elements of this axis

    Note that any local tensor can be indexed with (global_offset,
    local_offset, local_shape) if the data exists on this rank.
    """
    param_id: int = field(default=None)
    dtype: torch.dtype = field(default=None)

    local_shape: Tuple[int, ...] = field(default=None)
    global_shape: Tuple[int, ...]= field(default=None)

    axis_fragmentations: Tuple[int, ...]= field(default=None)

    local_offset: Tuple[int, ...] = field(default=None)
    global_offset: Tuple[int, ...] = field(default=None)

    def copy(self):
        return deepcopy(self)

    def fragment(self, num_frags: int, axis: int=0) -> List['ShardedTensorInfo']:
        """Apply new num_frags on the given axis. This operation
        may return one or more shards if the origin shard cannot
        be placed in one local tensor under new num_frags.

        Returns:
            List[ShardedTensorInfo]: a list of new shards
        """
        if self.global_shape[axis] % num_frags != 0:
            raise ValueError(f"Cannot fragment {self} with {num_frags} fragments.")
        offset = (
            self.local_offset[axis] +
            (
                self.global_offset[axis] *
                self.global_shape[axis] //
                self.axis_fragmentations[axis]
            )
        )
        stride = self.global_shape[axis] // num_frags
        new_shards = []
        for frag_idx in range(num_frags):
            cur = frag_idx * stride
            if cur + stride <= offset or cur >= offset + self.local_shape[axis]:
                continue
            left = max(cur, offset)
            right = min(cur + stride, offset + self.local_shape[axis])
            new_shards.append(ShardedTensorInfo(
                param_id=self.param_id,
                dtype=self.dtype,
                local_shape=self.local_shape[:axis] + (right - left, ) + self.local_shape[axis + 1:],
                global_shape=self.global_shape,
                axis_fragmentations=self.axis_fragmentations[:axis] + (num_frags, ) + self.axis_fragmentations[axis + 1:],
                local_offset=self.local_offset[:axis] + (left - cur, ) + self.local_offset[axis + 1:],
                global_offset=self.global_offset[:axis] + (frag_idx, ) + self.global_offset[axis + 1:]
            ))
        return new_shards

    def refragment(self, num_frags: int, axis: int=0) -> 'ShardedTensorInfo':
        """An wrapper to fragment with the given num_frags
        which has been applied earlier. In this case, the
        length of returned shards is always 1.

        Args:
            num_frags (int): The given num_frags.
            axis (int, optional): The given axis. Defaults to 0.

        Returns:
            ShardedTensorInfo: The new shard.
        """
        shard = self.fragment(num_frags, axis)
        if len(shard) != 1:
            raise ValueError(
                f"Try to refragment {self} with num_frags {num_frags} but got two or more shards."
            )
        return shard[0]

    def map_to_frag_id(self, frag_idx: int, axis: int=0) -> 'ShardedTensorInfo':
        """Return a new shard with different frag_idx
        on the given axis. Used when you want to index
        another local tensor with the same local offset.

        Args:
            frag_idx (int): The new frag_idx
            axis (int, optional): The given axis. Defaults to 0.

        Returns:
            ShardedTensorInfo: The new shard
        """
        if frag_idx < 0 or self.axis_fragmentations[axis] <= frag_idx:
            raise ValueError(f"Invalid frag_idx {frag_idx} w.r.t. the shard {self}")
        copied = self.copy()
        copied.global_offset = self.global_offset[:axis] + (frag_idx, ) + self.global_offset[axis + 1:]
        return copied

    def unsqueeze(self, offset:int, length: int, axis: int=0) -> 'ShardedTensorInfo':
        """Unsqueeze the sharded info on the given axis

        Args:
            offset (int): The new offset on the given axis
            length (int): The length of the new axis
            axis (int, optional): The axis to be unsqueezed. Defaults to 0.

        Returns:
            ShardedTensorInfo: The unsqueezed shard
        """
        return ShardedTensorInfo(
            param_id=self.param_id,
            dtype=self.dtype,
            local_shape=self.local_shape[:axis] + (1, ) + self.local_shape[axis:],
            global_shape=self.global_shape[:axis] + (length, ) + self.global_shape[axis:],
            axis_fragmentations=self.axis_fragmentations[:axis] + (length, ) + self.axis_fragmentations[axis:],
            global_offset=self.global_offset[:axis] + (offset, ) + self.global_offset[axis:],
            local_offset=self.local_offset[:axis] + (0, ) + self.local_offset[axis:]
        )

    def index(self, tensor: torch.Tensor) -> torch.Tensor:
        """Indexing tensor with this ShardedTensorInfo.
        will check the shape-related information and ignore 
        inconsistent datatype.
        
        Args:
            tensor (torch.Tensor): tensor to be indexed.
        

        """
        tensor_shape = tensor.shape
        local_tensor_shape = tuple(g // a for g, a in zip(self.global_shape, self.axis_fragmentations))
        if len(tensor.shape) != self.ndim or tensor_shape != local_tensor_shape:
            raise ValueError(
                f"Try to index a tensor with shape {tensor_shape} with a ShardedTensorInfo with shape {local_tensor_shape}."
            )
        indices = [slice(offset, offset + length) for offset, length in zip(self.local_offset, self.local_shape)]
        view = tensor[indices]
        if view.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr():
            # NOTE: Ensure that we return the view
            raise SystemError("Unexpected copy operation encountered!")
        return view


    @classmethod
    def from_global_shape(cls, global_shape: Tuple[int, ...], param_id: int=None, dtype: torch.dtype=None):
        return cls(
            param_id=param_id,
            dtype=dtype,
            local_shape=global_shape,
            global_shape=global_shape,
            axis_fragmentations=(1, ) * len(global_shape),
            local_offset=(0, ) * len(global_shape),
            global_offset=(0, ) * len(global_shape)
        )

    @property
    def ndim(self):
        return len(self.global_shape)

    @property
    def size(self):
        assert self.dtype is not None, "Only support shard_info with dtype"
        return np.prod(self.local_shape) * self.dtype.itemsize

    def numel(self):
        return np.prod(self.local_shape)

    def __hash__(self):
        return hash((
            self.param_id,
            self.dtype,
            *self.local_shape,
            *self.global_shape,
            *self.axis_fragmentations,
            *self.local_offset,
            *self.global_offset
        ))

    def __post_init__(self):
        assert self.axis_fragmentations is not None
        assert self.global_offset is not None
        assert self.global_shape is not None

        if self.local_offset is None:
            self.local_offset = (0, ) * self.ndim

        assert (
            min(self.axis_fragmentations) > 0 and
            min(self.global_offset) >= 0 and
            min(self.local_offset) >= 0
        )
        assert all(l % a == 0 for l, a in zip(self.global_shape, self.axis_fragmentations))
        assert all(s < a for s, a in zip(self.global_offset, self.axis_fragmentations))

        grid_shape = tuple(l // a for l, a in zip(self.global_shape, self.axis_fragmentations))
        if self.local_shape is not None:
            assert min(self.local_shape) >= 0
            assert all(
                s + l <= g for s, l, g in zip(
                    self.local_offset,
                    self.local_shape,
                    grid_shape
                )
            )
        else:
            assert all(self.local_offset) == 0
            self.local_shape = grid_shape

    @staticmethod
    def concat(shards: List['ShardedTensorInfo'], axis: int=0) -> Optional['ShardedTensorInfo']:
        """Merge some continuous shards into one on the given axis.


        Returns:
            ShardedTensorInfo: The concatenated shard. If the input list is empty, 
            returns None.
        """
        if len(shards) == 0:
            return None

        if len(shards) == 1:
            return shards[0].copy()
        def get_values(shards: List['ShardedTensorInfo'], key: str):
            return [getattr(shard, key) for shard in shards]

        def all_equal(shards: List['ShardedTensorInfo'], key: str):
            return all(getattr(shard, key) == getattr(shards[0], key) for shard in shards)

        def all_equal_except_axis(shards: List['ShardedTensorInfo'], key: str, axis: int):
            return all(
                getattr(shard, key)[:axis] == getattr(shards[0], key)[:axis] and
                getattr(shard, key)[axis+1:] == getattr(shards[0], key)[axis+1:]
                for shard in shards
            )

        for key in ['global_shape', 'param_id', 'global_offset', 'dtype']:
            if not all_equal(shards, key):
                raise ValueError(
                    f"Concat on shards with inconsistent {key} is "
                    f"invalid, got: {get_values(shards, key)}."
                )

        for key in ['local_shape', 'local_offset']:
            if not all_equal_except_axis(shards, key, axis):
                raise ValueError(
                    f"Concat on shards with inconsistent {key} on non-target axis is "
                    f"invalid, got: {get_values(shards, key)}."
                )

        shards = sorted(shards, key=lambda x: x.local_offset[axis])
        left = shards[0].local_offset[axis]
        right = shards[-1].local_offset[axis] + shards[-1].local_shape[axis]
        return ShardedTensorInfo(
            param_id=shards[0].param_id,
            dtype=shards[0].dtype,
            local_shape=shards[0].local_shape[:axis] + (right - left,) + shards[0].local_shape[axis + 1:],
            global_shape=shards[0].global_shape,
            axis_fragmentations=shards[0].axis_fragmentations,
            local_offset=shards[0].local_offset,
            global_offset=shards[0].global_offset
        )

    def __contains__(self, other: 'ShardedTensorInfo'):
        """Check if the other sharded_info is in this sharded_info.
        Sharded_info A is in B only if B has the same global metadata as A,
        and the local shard of B is a sub-shard of A.

        Args:
            other (ShardedTensorInfo): _description_

        Returns:
            _type_: _description_
        """
        if other.param_id is not None and other.param_id != self.param_id:
            return False

        if other.dtype is not None and other.dtype != self.dtype:
            return False

        if (
            other.global_shape != self.global_shape or
            other.axis_fragmentations != self.axis_fragmentations or
            other.global_offset != self.global_offset
        ):
            return False

        for si, oi, sj, oj in zip(
            self.local_offset,
            other.local_offset,
            self.local_shape,
            other.local_shape
        ):
            if si > oi or si + sj < oi + oj:
                return False
        return True
