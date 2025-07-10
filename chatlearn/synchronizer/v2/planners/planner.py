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
"""Sync parameters"""
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
import concurrent.futures
import traceback
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import List
from queue import PriorityQueue

import torch
from tqdm import tqdm

from chatlearn.launcher.initialize import patch_ray
from chatlearn.utils import future
from chatlearn.utils import utils
from chatlearn.utils.constant import PARAM_SYNC_COMM_TYPE
from chatlearn.utils.constant import ROUTED_EXPERT_REGROUPING_COMM_TYPE
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.utils.utils import execute_in_parallel
from chatlearn.utils.timer import Timers
from chatlearn.utils.mappings import ShardedTensorInfo

patch_ray()

from typing import *

@dataclass(frozen=True)
class Ranks:
    values: tuple[int, ...]
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

# NOTE: some methods are actually for general use, extract a base class if needed.
class MegatronVLLMSyncPlanner:
    """Generate the sync plan based on the given sync mapping and 
    ModelParallel setting. The plan is a mapping of send(recv) sharded 
    parameters in one iteration.
    """
    def __init__(
        self, 
        sync_mapping: Dict[int, Dict[ShardedTensorInfo, List[ShardedTensorInfo]]], 
        dst_metadata: Dict[int, List[ShardedTensorInfo]],
        bucket_size: int = 4 * 1024 ** 3,
    ):
        """The sync planner for Megatron to vLLM to balance the payloads
        and minimize the communication cost.

        Args:
            sync_mapping (Dict[int, Dict[ShardedTensorInfo, List[ShardedTensorInfo]]]): 
            The mapping information from all source ranks.
            dst_metadata (Dict[int, List[ShardedTensorInfo]]): The parameter metadata
            of the destination model.
        """
        self.sync_mapping = sync_mapping
        self.dst_metadata = dst_metadata
        self.bucket_size = bucket_size

        self._bucket_id = 0

        # NOTE: mapping an source weight to its owners and destination weights.
        # In some cases, a source weight may be mapped to multiple destination weights.
        self.src_param_to_src_ranks = defaultdict(list)
        self.src_param_to_dst_params = defaultdict(set)
        for rank, sync_mapping in self.sync_mapping.items():
            for src_param, dst_params in sync_mapping.items():
                self.src_param_to_src_ranks[src_param].append(rank)
                self.src_param_to_dst_params[src_param].update(dst_params)
        
        # NOTE: mapping an dst weight to its owners
        self.dst_param_to_dst_ranks = defaultdict(list)
        for rank, metadata in self.dst_metadata.items():
            for param in metadata:
                self.dst_param_to_dst_ranks[param].append(rank)
        
        # NOTE: mapping an src weight to its destinations, be cautious that
        # each destination may have multiple weights requiring the same 
        # source weight. (to be checked)
        self.src_param_to_dst_ranks = defaultdict(set)
        for src_param, dst_params in self.src_param_to_dst_params.items():
            for dst_param in dst_params:
                for target, dst_ranks in self.dst_param_to_dst_ranks.items():
                    if dst_param in target:
                        self.src_param_to_dst_ranks[src_param].update(dst_ranks)

    def make_plan(self) -> Tuple[Dict[int, List[SyncIteration]], ...]:
        """
            Make a general plan, do not care the specifc model parallel
            information of the source or dst model.
        """
        # NOTE: The sender in one sender_groups can send the same set of 
        # parameters, thus we can balance payloads in this group
        sender_groups = defaultdict(list)
        for src_param, src_ranks in self.src_param_to_src_ranks.items():
            sender_groups[Ranks(src_ranks)].append(src_param)

        # type: Dict[int, Dict[Ranks, List[ShardedTensorInfo]]]
        unbucketized_plan = defaultdict(lambda: defaultdict(list))
        for sender_group, all_payloads in sender_groups.items():
            items = np.array([shard.size for shard in all_payloads])
            # NOTE: as we cannot determine the number of iters in advance,
            # only thing we can do is to balance in the sender_group
            balanced_payloads = self.approximate_bin_packing(items, len(sender_group.values))
            for rank, payload_indices in zip(sender_group.values, balanced_payloads):
                for idx in payload_indices:
                    unbucketized_plan[rank][
                        Ranks(tuple(self.src_param_to_dst_ranks[all_payloads[idx]]))
                    ].append(all_payloads[idx])
        
        # TODO: collect the bucket statistics of each rank

        # Bucketize on each rank
        bucketized_plan = self.bucketize(unbucketized_plan, bucket_size=self.bucket_size)

        # NOTE: Convert to logical iterations, which means each rank
        # will know what it will send/recv to others. In ChatLearn,
        # different models are on different actors, thus each SyncIteration
        # will has an empty send_buckets or recv_buckets
        sender_plan, recver_plan = self.convert_to_iterations(bucketized_plan)

        # TODO: collect the iteration statistics of each rank
        return sender_plan, recver_plan
    
    def setup_synchronizer(
        self, 
        src_model: 'DistModel', 
        dst_model: 'DistModel', 
        sender_plan: Dict[int, List[SyncIteration]], 
        recver_plan: Dict[int, List[SyncIteration]],
        group_name: str = "default_sync_group",
    ):
        """Setup synchronizer for the actual parameter synchronization.
        
        
        """
        sender_plan, recver_plan = deepcopy(sender_plan), deepcopy(recver_plan)

        # NOTE: find colocated actors between src_model and dst_model
        src_gpus = future.wait(src_model.call_func_on_all_workers('get_gpu_info'), return_output=True)
        dst_gpus = future.wait(dst_model.call_func_on_all_workers('get_gpu_info'), return_output=True)

        # TODO: `get_rank` will get base_module._rank instead of dist.get_rank()
        comm_ranks = future.wait(src_model.call_func_on_all_workers('get_torchdist_rank'), return_output=True)

        src_rank_to_gpu_id = dict(zip(itertools.chain.from_iterable(src_model.all_ranks), src_gpus))
        gpu_id_to_dst_rank = dict(zip(dst_gpus, itertools.chain.from_iterable(dst_model.all_ranks)))

        if set(src_gpus) != set(dst_gpus):
            raise NotImplementedError(
                f'The source and destination model in partial colocated/ no colocated mode is not supported. '
            )

        # replace recver_ranks in all send_buckets with colocated MCore comm rank
        dst_rank_to_gpu_id = dict(zip(itertools.chain.from_iterable(dst_model.all_ranks), dst_gpus))
        gpu_id_to_src_rank = dict(zip(src_gpus, itertools.chain.from_iterable(src_model.all_ranks)))
        for rank, send_iteration in sender_plan.items():
            for send_bucket, recv_ranks in send_iteration.send_buckets.items():
                # dst_rank -> gpu_id -> src_rank -> comm_rank
                src_ranks = [gpu_id_to_src_rank[dst_rank_to_gpu_id[rank]] for rank in recv_ranks]
                send_iteration.send_buckets[send_bucket] = Ranks(
                    [comm_ranks[src_rank] for src_rank in src_ranks]
                )

        # replace sender_rank in all recv_buckets with MCore comm rank
        for rank, recv_iteration in recver_plan.items():
            for recv_bucket, src_rank in recv_iteration.recv_buckets.items():
                recv_iteration.recv_buckets[recv_bucket] = comm_ranks[src_rank]

        refs = []
        for src_rank, src_gpu_id in src_rank_to_gpu_id.items():
            dst_rank = gpu_id_to_dst_rank[src_gpu_id]
            for send_iteration, recv_iteration in zip(
                sender_plan[src_rank],
                recver_plan[dst_rank]
            ):
                # NOTE: MCore actors perform all2all should know both 
                # send_buckets/recv_buckets
                send_iteration.recv_buckets = recv_iteration.recv_buckets

            src_actor = src_model.get_actor(src_rank)
            refs.append(src_actor.set_synchronizer(
                synchronizer_name = 'general',
                return_handles = True,
                local_plan = sender_plan[src_rank],
                synchronizer_type = SynchronizerType.SEND,
                in_process_group = True
            ))
        
        handles = future.wait(refs, return_output=True)
        # NOTE: setup synchronizer of dst actors, assign `param_provider` handle
        refs = []
        for src_rank, src_gpu_id in src_rank_to_gpu_id.items():
            dst_rank = gpu_id_to_dst_rank[src_gpu_id]
            dst_actor = dst_model.get_actor(dst_rank)
            refs.append(dst_actor.set_synchronizer(
                synchronizer_name = 'general',
                return_handles = False,
                local_plan = recver_plan[dst_rank],
                synchronizer_type = SynchronizerType.SEND,
                in_process_group = False,
                sync_step_handle = handles[src_rank][0],
                release_ipc_handle = handles[src_rank][1]
            ))
        future.wait(refs, return_output=True)

    @staticmethod
    def approximate_bin_packing(items: np.array, K: int) -> List[List[int]]:
        """Packing N items into K bins and make the payloads 
            of each bin as close as possible.

        Args:
            items (np.array): The sizes of each item
            K (int): The num of buckets.

        Returns:
            List[List[int]]: The item index of each bucket.
        """
        bins = np.zeros(K)
        results = [list() for _ in range(K)]
        for idx in items.argsort()[::-1]:
            bins[bins.argmin()] += items[idx]
            results[bins.argmin()].append(idx)
        return results
    
    def bucketize(
        self,
        unbucketized_plan: Dict[int, Dict[Ranks, List[ShardedTensorInfo]]],
        bucket_size: int = 4 * 1024 ** 3,
    ) -> Dict[int, Dict[Ranks, List[BucketInfo]]]:
        """Bucketize the unbucketized plan. Each `List[ShardedTensorInfo]`
        will be chunked into several buckets according to the bucket size.

        Args:
            unbucketized_plan (Dict[int, Dict[Ranks, List[ShardedTensorInfo]]]): 
            The unbucketized plan.
            bucket_size (int, optional): The size of each bucket. Defaults to 
            `4 * 1024 ** 3`.

        Returns:
            Dict[int, Dict[Ranks, List[BucketInfo]]]: The bucketized plan.
        """
        def create_bucket(shards: List[ShardedTensorInfo]) -> BucketInfo:
            send_layout = []
            recv_layout = []
            offset = 0
            for shard in shards:
                send_layout.append((offset, shard.copy()))
                for dst_param in self.src_param_to_dst_params[shard]:
                    recv_layout.append((offset, dst_param.copy()))
                offset += shard.size
            ret = BucketInfo(
                bucket_id=self._bucket_id,
                send_layout=send_layout,
                recv_layout=recv_layout
            )
            self._bucket_id += 1
            return ret

        bucketized_plan = defaultdict(lambda: defaultdict(list))
        for sender_rank, plan_per_rank in unbucketized_plan.items():
            for receiver_ranks, shard_info_list in plan_per_rank.items():
                current_size = 0
                shards = []
                for shard in shard_info_list:
                    if current_size + shard.size > bucket_size:
                        bucketized_plan[sender_rank][receiver_ranks].append(
                            create_bucket(shards)
                        )
                        current_size = 0
                        shards = []
                    current_size += shard.size
                    shards.append(shard)
                if len(shards) > 0:
                    bucketized_plan[sender_rank][receiver_ranks].append(
                        create_bucket(shards)
                    )

        return bucketized_plan

    def convert_to_iterations(
        self, 
        bucketized_plan: Dict[int, Dict[Ranks, List[BucketInfo]]],
        max_buckets_per_iteration: int=-1
    ) -> Tuple[Dict[int, List[SyncIteration]], ...]:
        """Given a bucketized plan, convert it to a list of iterations.
        
        Returns:
            sender_plan (Dict[int, List[SyncIteration]]): The iterations 
            for each rank of the sender.
            recver_plan (Dict[int, List[SyncIteration]]): The iterations
            for each rank of the receiver.
        """
        # NOTE: for each rank, attempt to send buckets as much as 
        # possible in one iteration. Bucket A and B can be parallelized
        # only if receivers of A and B are non-overlapped. Greedy is adopted
        # here to maximize the bucket amounts.
        sender_to_iterations = {}
        for sender_rank, buckets in bucketized_plan.items():
            bucket_to_recvers = {b: k for k, v in buckets.items() for b in v}
            sorted_buckets = sorted(bucket_to_recvers.items(), key=lambda x: x[0].size, reverse=True)
            assert len(set(bucket_to_recvers)) == len(bucket_to_recvers), "Buckets are not unique"
            is_added = set()
            iterations = []
            while len(is_added) < len(sorted_buckets):
                this_iteration = []
                is_busy = set()
                for bucket, recvers in sorted_buckets:
                    if bucket in is_added:
                        continue
                    if all(r not in is_busy for r in recvers.values):
                        this_iteration.append((bucket, recvers))
                        is_busy.update(recvers.values)
                        is_added.add(bucket)
                iterations.append(this_iteration)
            sender_to_iterations[sender_rank] = iterations

        n_iterations = max(len(v) for v in sender_to_iterations.values())
        sender_plan = defaultdict(list)
        recver_plan = defaultdict(lambda: [SyncIteration() for _ in range(n_iterations)])
        for sender_rank, iterations in sender_to_iterations.items():
            for iter_idx in range(n_iterations):
                this_iteration = iterations[iter_idx] if iter_idx < len(iterations) else []
                sender_plan[sender_rank].append(SyncIteration(
                    send_buckets=dict(this_iteration),
                    recv_buckets={},
                ))

                for bucket, recvers in this_iteration:
                    for recver_rank in recvers.values:
                        recver_plan[recver_rank][iter_idx].recv_buckets[bucket] = sender_rank
        return sender_plan, recver_plan

    def _find_dst_ranks(self, sharded_info: ShardedTensorInfo):
        """Find dst ranks containing this sharded info"""
        for param, ranks in self.dst_param_to_ranks:
            if sharded_info in param:
                return ranks