from typing import *
from enum import Enum
from dataclasses import dataclass, field
from copy import deepcopy

import torch

from chatlearn.utils.mappings import ShardedTensorInfo


class SynchronizerType(Enum):
    SEND = 0
    RECV = 1
    FORWARD = 2

@dataclass(frozen=True)
class Ranks:
    values: Tuple[int, ...]
    def __post_init__(self):
        object.__setattr__(self, 'values', tuple(sorted(self.values)))
    def __hash__(self):
        return hash(self.values)


@dataclass
class BucketInfo:
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