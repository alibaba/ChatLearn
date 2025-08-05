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
"""structs for parameter synchronization"""
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field
from copy import deepcopy

import torch

from chatlearn.utils.mappings import ShardedTensorInfo


class SynchronizerType(Enum):
    """The type of synchronizers, currently FORWARD is not used."""
    SEND = 0
    RECV = 1
    FORWARD = 2

@dataclass(frozen=True)
class Ranks:
    """An ordered tuple of ranks. We expect Ranks((1, 3, 2)) to be 
    equal to Ranks((1, 2, 3))
    """
    values: Tuple[int, ...]
    def __post_init__(self):
        object.__setattr__(self, 'values', tuple(sorted(self.values)))
    def __hash__(self):
        return hash(self.values)


@dataclass
class BucketInfo:
    """The bucket of some parameters to be synchronized.
    The offset and sharded_info.size of `send_layout` and `recv_layout` 
    indicates how parameters locate (or will locate if the buffer is None) 
    on the buffer in this bucket.
    """
    bucket_id: int
    # offset in bucket, sharded_info
    send_layout: List[Tuple[int, ShardedTensorInfo]]
    # offset in bucket, sharded_info
    recv_layout: List[Tuple[int, ShardedTensorInfo]]
    buffer: Optional[torch.Tensor] = None
    size: int = None

    def __hash__(self):
        return hash(self.bucket_id)

    def copy(self):
        return deepcopy(self)

    def __post_init__(self):
        self.size = sum(info.size for _, info in self.send_layout)

@dataclass
class SyncIteration:
    """One iteration of plan for some rank"""
    # send bucket to a list of remote ranks
    send_buckets: Dict[BucketInfo, Ranks] = field(default_factory=dict)
    # recv bucket from one remote rank
    recv_buckets: Dict[BucketInfo, int] = field(default_factory=dict)
