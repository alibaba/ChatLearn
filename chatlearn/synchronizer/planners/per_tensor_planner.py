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
from itertools import chain
from typing import Dict, List, Tuple, TYPE_CHECKING, Any

import numpy as np

from chatlearn.utils import future
from chatlearn.utils.logger import logger
from chatlearn.utils.timer import Timers
from chatlearn.utils.mappings import ShardedTensorInfo
from chatlearn.synchronizer.structs import (
    SynchronizerType,
    Ranks,
    BucketInfo,
    SyncIteration
)

from .base_planner import BasePlanner
if TYPE_CHECKING:
    from chatlearn.runtime.dist_actor import DistModel



class PerTensorPlanner(BasePlanner):
    """Generate a plan that ensure each dst parameter will be synchronized in one
    iteration.
    """

    def setup_synchronizer(
        self,
        src_model: 'DistModel',
        dst_model: 'DistModel',
        plan: Any,
    ):
        """Generate the final synchronization plan and setup synchronizer
        for the actual parameter synchronization.

        Args:
            src_model (DistModel): The source dist model to be synchronized.
            dst_model (DistModel): The dst dist model to be synchronized.
            plan (...): The plan returned by `self.make_plan()`
        """
        # NOTE: find colocated actors between src_model and dst_model
        self.timers("prepare-metadata").start()
        src_gpus = future.wait(src_model.call_func_on_all_workers('get_gpu_info'), return_output=True)
        dst_gpus = future.wait(dst_model.call_func_on_all_workers('get_gpu_info'), return_output=True)
        if set(src_gpus) != set(dst_gpus):
            raise NotImplementedError(
                'The source and destination model in partial colocated/ no colocated mode is not supported. '
            )
        comm_ranks = future.wait(src_model.call_func_on_all_workers('get_rank'), return_output=True)

        src_rank_to_gpu_id = dict(zip(chain.from_iterable(src_model.all_actor_ids), src_gpus))
        dst_rank_to_gpu_id = dict(zip(chain.from_iterable(dst_model.all_actor_ids), dst_gpus))
        gpu_id_to_dst_rank = dict(zip(dst_gpus, chain.from_iterable(dst_model.all_actor_ids)))
        gpu_id_to_src_rank = dict(zip(src_gpus, chain.from_iterable(src_model.all_actor_ids)))

        sender_plan, recver_plan = plan

        # replace recver_ranks in all send_buckets with colocated MCore comm rank
        for rank, send_iterations in sender_plan.items():
            for send_iteration in send_iterations:
                for send_bucket, recv_ranks in send_iteration.send_buckets.items():
                    # dst_rank -> gpu_id -> src_rank -> comm_rank
                    src_ranks = [gpu_id_to_src_rank[dst_rank_to_gpu_id[rank]] for rank in recv_ranks.values]
                    send_iteration.send_buckets[send_bucket] = Ranks(
                        [comm_ranks[src_rank] for src_rank in src_ranks]
                    )

        # replace sender_rank in all recv_buckets with MCore comm rank
        for rank, recv_iterations in recver_plan.items():
            for recv_iteration in recv_iterations:
                for recv_bucket, src_rank in recv_iteration.recv_buckets.items():
                    recv_iteration.recv_buckets[recv_bucket] = comm_ranks[src_rank]

        # NOTE: finally, register synchronizer for each actor
        self.timers("register-synchronizer").start()
        self._register_synchronizer(
            src_model,
            dst_model,
            src_rank_to_gpu_id,
            gpu_id_to_dst_rank,
            sender_plan,
            recver_plan,
        )
        self.timers("register-synchronizer").stop()
        logger.info(f"finish setup synchronizer | {self.timers.log()}")

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
    ) -> Tuple[Dict[int, List[SyncIteration]], Any]:
        """Given a bucketized plan, convert it to a list of iterations.

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
        
        src_shard_to_sender = dict()
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

    def _register_synchronizer(
        self,
        src_model,
        dst_model,
        src_rank_to_gpu_id,
        gpu_id_to_dst_rank,
        sender_plan,
        recver_plan,
    ):
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
            refs.append(src_actor.set_synchronizer.remote(
                synchronizer_name = 'general',
                local_plan = sender_plan[src_rank],
                synchronizer_type = SynchronizerType.SEND,
                in_process_group = True
            ))
        future.wait(refs, return_output=True)
        # NOTE: setup synchronizer of dst actors, assign `param_provider` handle
        refs = []
        for src_rank, src_gpu_id in src_rank_to_gpu_id.items():
            dst_rank = gpu_id_to_dst_rank[src_gpu_id]
            dst_actor = dst_model.get_actor(dst_rank)
            refs.append(dst_actor.set_synchronizer.remote(
                synchronizer_name = 'general',
                local_plan = recver_plan[dst_rank],
                synchronizer_type = SynchronizerType.RECV,
                in_process_group = False,
                colocate_handle = src_model.get_actor(src_rank).call_synchronizer_func
            ))
        future.wait(refs, return_output=True)

