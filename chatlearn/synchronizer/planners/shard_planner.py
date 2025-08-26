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
from collections import defaultdict
from typing import Dict, List, Tuple, TYPE_CHECKING

from chatlearn.utils.mappings import ShardedTensorInfo
from chatlearn.synchronizer.structs import (
    Ranks,
    BucketInfo,
    SyncIteration
)
from .base_planner import BasePlanner

if TYPE_CHECKING:
    from chatlearn.runtime.dist_actor import DistModel


class ShardwiseSyncPlanner(BasePlanner):
    """Generate the sync plan based on the given sync mapping. The plan is 
    a mapping of send(recv) sharded parameters in iterations. 
    """

    def _convert_to_iterations(
        self,
        bucketized_plan: Dict[int, Dict[Ranks, List[BucketInfo]]],
        src_rank_to_gpu_id: Dict[int, int],
        dst_rank_to_gpu_id: Dict[int, int],
        mem_infos: Dict[int, Tuple[int, int]],
        max_memory_fraction: float=0.8
    ) -> Tuple[Dict[int, List[SyncIteration]], ...]:
        """Given a bucketized plan, convert it to a list of iterations.

        Args:
            bucketized_plan (Dict[int, Dict[Ranks, List[BucketInfo]]]): 
            The bucketized plan returned by `bucketize`.
            src_rank_to_gpu_id (Dict[int, int]): map ranks of source model to 
            physical GPU ID.
            dst_rank_to_gpu_id (Dict[int, int]): map ranks of destination model 
            to physical GPU ID.
            mem_infos (Dict[int, Tuple[int, int]]): The used memory and 
            total memory for each physical GPU.
            max_memory_fraction (float, optional): The maximum ratio of planner 
            could use. Defaults to 0.8.

        Returns:
            sender_plan (Dict[int, List[SyncIteration]]): The plan of senders.
            receiver_plan (Dict[int, List[SyncIteration]]): The plan of receivers.
        """
        # NOTE: for each rank, attempt to send buckets as much as
        # possible in one iteration. Bucket A and B can be parallelized
        # only if receivers of A and B are non-overlapped. Greedy is adopted
        # here to maximize the bucket amounts.
        budgets = {
            gpu_id: mem_info[0] - (1 - max_memory_fraction) * mem_info[1]
            for gpu_id, mem_info in mem_infos.items()
        }

        flatten_buckets = deepcopy([
            (sender, bucket, receiver)
            for sender, receivers_to_buckets in bucketized_plan.items()
            for receiver, buckets in receivers_to_buckets.items()
            for bucket in buckets
        ])
        # prefer bucket with larger bucket_size * n_receivers
        flatten_buckets.sort(key=lambda x: x[1].size * len(x[2].values), reverse=True)
        sender_to_iterations = defaultdict(list)
        is_added = set()
        n_iterations = 0

        while len(is_added) < len(flatten_buckets):
            budgets_this_iter = deepcopy(budgets)
            is_busy = set()
            n_iterations += 1
            for sender in bucketized_plan:
                sender_to_iterations[sender].append([])

            is_progressed = False
            for sender, bucket, receivers in flatten_buckets:
                if bucket in is_added:
                    continue
                # NOTE: calc required memory, each sender and receiver will
                # alloc a bucket, if sender and receiver is colocated, only one
                # bucket is needed on the colocated gpu
                required_mem = {k: 0 for k in budgets_this_iter}
                for receiver in receivers.values:
                    required_mem[dst_rank_to_gpu_id[receiver]] += bucket.size
                if sender not in receivers.values:
                    required_mem[src_rank_to_gpu_id[sender]] += bucket.size

                if any(required_mem[k] > budgets_this_iter[k] for k in required_mem):
                    continue
                if all((sender, receiver) not in is_busy for receiver in receivers.values):
                    is_progressed = True
                    is_added.add(bucket)

                    sender_to_iterations[sender][-1].append((bucket, receivers))
                    for receiver in receivers.values:
                        is_busy.add((sender, receiver))
                    for gpu_id in budgets_this_iter:
                        budgets_this_iter[gpu_id] -= required_mem[gpu_id]

            if not is_progressed:
                raise RuntimeError(
                    f"Not enough memory to apply the plan with max_memory_fraction {max_memory_fraction}."
                )

        # NOTE: format the plan with SyncIteration dataclass
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

    def build_iteration(
        self,
        unbucketized_plan: Dict[int, Dict[Ranks, List[ShardedTensorInfo]]],
        src_rank_to_gpu_id: Dict[int, int],
        dst_rank_to_gpu_id: Dict[int, int],
        mem_infos: Dict[int, Tuple[int, int]],
        max_memory_fraction: float=0.8
    ) -> List[Dict[int, List[SyncIteration]]]:
        """Build iterations from unbucketized plan according to the
        given memory constraints.
        
        Args:
            unbucketized_plan (Dict[int, Dict[Ranks, List[ShardedTensorInfo]]]): 
            The unbucketized comm plan.
            src_rank_to_gpu_id (Dict[int, int]): map ranks of source model to 
            physical GPU ID.
            dst_rank_to_gpu_id (Dict[int, int]): map ranks of destination model 
            to physical GPU ID.
            mem_infos (Dict[int, Tuple[int, int]]): The used memory and 
            total memory for each physical GPU.
            max_memory_fraction (float, optional): The maximum ratio of planner 
            could use. Defaults to 0.8.

        Returns:
            sender_plan (Dict[int, List[SyncIteration]]): The plan of senders.
            receiver_plan (Dict[int, List[SyncIteration]]): The plan of receivers.
        """
        bucketized_plan = self.bucketize(unbucketized_plan, bucket_size=self.bucket_size)
        return self._convert_to_iterations(
            bucketized_plan,
            src_rank_to_gpu_id,
            dst_rank_to_gpu_id,
            mem_infos,
            max_memory_fraction
        )
