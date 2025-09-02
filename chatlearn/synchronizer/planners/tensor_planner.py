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
    SyncIteration
)

from .base_planner import BasePlanner

if TYPE_CHECKING:
    from chatlearn.runtime.dist_actor import DistModel



class TensorwisePlanner(BasePlanner):
    """Generate a plan that ensure each dst parameter will be synchronized in one
    iteration.
    """

    def _check_constraint(
        self,
        shards,
        budgets_this_iter,
        src_shard_to_sender
    ):
        allocated_bucket_size = deepcopy(self.allocated_bucket_size)
        required_mem = {k: 0 for k in budgets_this_iter}
        for shard in shards:
            sender = src_shard_to_sender[shard]
            receivers = Ranks(tuple(self.src_param_to_dst_ranks[shard]))

            for receiver in receivers.values:
                # NOTE: Per-Tensor-Sync requires to merge shards of weights on receivers,
                # leading to an extra copy
                required_mem[self.dst_rank_to_gpu_id[receiver]] += 2 * shard.size

            # NOTE: 1. sender can send two shards only if receivers of these two shards are same.
            if (
                any((sender, recver) in self.is_busy for recver in receivers.values) and
                receivers not in self.receiver_group[sender]
            ):
                return False, None

            # NOTE: 2. The bucket for the receiver group cannot exceed bucket size limit
            if allocated_bucket_size[sender][receivers] + shard.size > self.bucket_size:
                return False, None

            allocated_bucket_size[sender][receivers] += shard.size
            if sender not in receivers.values:
                required_mem[self.src_rank_to_gpu_id[sender]] += shard.size

        if not all(required_mem[k] <= budgets_this_iter[k] for k in required_mem):
            return False, None

        self.allocated_bucket_size = allocated_bucket_size
        return True, required_mem

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
        self.budgets = {
            gpu_id: mem_info[0] - (1 - max_memory_fraction) * mem_info[1]
            for gpu_id, mem_info in mem_infos.items()
        }
        self.src_rank_to_gpu_id = src_rank_to_gpu_id
        self.dst_rank_to_gpu_id = dst_rank_to_gpu_id

        dst_param_id_to_src_params: Dict[int, List[ShardedTensorInfo]] = defaultdict(list)
        for src_param, dst_params in self.src_param_to_dst_params.items():
            is_added = set()
            for dst_param in dst_params:
                if dst_param.param_id in is_added:
                    continue
                is_added.add(dst_param.param_id)
                dst_param_id_to_src_params[dst_param.param_id].append(src_param)

        src_shard_to_sender = {}
        for sender, plan_per_rank in unbucketized_plan.items():
            for shards in plan_per_rank.values():
                for shard in shards:
                    if shard in src_shard_to_sender and sender != src_shard_to_sender[shard]:
                        print(f'find multiple sender for shard {shard}', flush=True)
                    src_shard_to_sender[shard] = sender

        is_added = set()
        n_iterations = 0
        collected_shards_list = []
        while len(is_added) < len(dst_param_id_to_src_params):
            n_iterations += 1
            budgets_this_iter = deepcopy(self.budgets)
            self.is_busy = set()
            self.receiver_group = defaultdict(set)
            self.allocated_bucket_size = defaultdict(lambda: defaultdict(int))
            collected_shards: Dict[int, Dict[Ranks, List[ShardedTensorInfo]]] = defaultdict(lambda: defaultdict(list))

            is_progressed = True
            while is_progressed:
                is_progressed = False
                for dst_param_id in dst_param_id_to_src_params:
                    if dst_param_id in is_added:
                        continue
                    src_shards = dst_param_id_to_src_params[dst_param_id]
                    can_be_added, required_mem = self._check_constraint(
                        src_shards,
                        budgets_this_iter,
                        src_shard_to_sender
                    )

                    if not can_be_added:
                        continue

                    is_progressed = True
                    is_added.add(dst_param_id)
                    for gpu_id in budgets_this_iter:
                        budgets_this_iter[gpu_id] -= required_mem[gpu_id]
                    for shard in src_shards:
                        sender = src_shard_to_sender[shard]
                        receivers = Ranks(tuple(self.src_param_to_dst_ranks[shard]))
                        # TODO: select one sender
                        for receiver in receivers.values:
                            self.is_busy.add((sender, receiver))
                        self.receiver_group[sender].add(receivers)
                        collected_shards[sender][receivers].append(shard)

            if len(self.is_busy) == 0:
                raise RuntimeError(
                    f"Not enough memory to apply the plan with max_memory_fraction {max_memory_fraction}."
                )
            collected_shards_list.append(collected_shards)
        return self._convert_to_iteration_format(collected_shards_list)

    def _convert_to_iteration_format(
        self,
        collected_shards_list: List[Dict[int, Dict[Ranks, List[ShardedTensorInfo]]]],
    ) -> Tuple[Dict[int, List[SyncIteration]], ...]:
        n_iterations = len(collected_shards_list)
        sender_plan = defaultdict(lambda: [SyncIteration() for _ in range(n_iterations)])
        recver_plan = defaultdict(lambda: [SyncIteration() for _ in range(n_iterations)])
        for iteration_idx, collected_shards in enumerate(collected_shards_list):
            for sender_rank, plan_per_rank in self.bucketize(
                collected_shards,
                bucket_size=self.bucket_size
            ).items():
                for receivers, buckets in plan_per_rank.items():
                    assert len(buckets) == 1, "Expect only one bucket for each receiver group"
                    sender_plan[sender_rank][iteration_idx].send_buckets[buckets[0]] = receivers
                    for receiver_rank in receivers.values:
                        recver_plan[receiver_rank][iteration_idx].recv_buckets[buckets[0]] = sender_rank

        return sender_plan, recver_plan
