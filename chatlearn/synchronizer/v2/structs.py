from typing import *
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
    # send bucket to a list of remote ranks
    send_buckets: Dict[BucketInfo, Ranks] = field(default_factory=dict) 
    # recv bucket from one remote rank
    recv_buckets: Dict[BucketInfo, int] = field(default_factory=dict) 