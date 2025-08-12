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

from chatlearn.launcher.initialize import patch_ray
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

patch_ray()

class MegatronVLLMSyncPlanner(BasePlanner):
    """Generate the sync plan based on the given sync mapping. The plan is 
    a mapping of send(recv) sharded parameters in iterations. 
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

        self.timers("prepare-metadata").stop()

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
    ):
        # NOTE: Convert to iterations, which means each rank will know how
        # it will send/recv to others. In ChatLearn, different models are
        # on different actors, thus each SyncIteration will either have an
        # empty send_buckets or recv_buckets
        bucketized_plan = self.bucketize(unbucketized_plan, bucket_size=self.bucket_size)
        return self._convert_to_iterations(
            bucketized_plan,
            src_rank_to_gpu_id,
            dst_rank_to_gpu_id,
            mem_infos,
            max_memory_fraction
        )

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
