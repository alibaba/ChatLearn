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
"""Sync parameters"""

import concurrent.futures
import traceback
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import List, Dict
from queue import PriorityQueue

import torch
from tqdm import tqdm

from chatlearn.launcher.initialize import patch_ray
from chatlearn.utils import future
from chatlearn.utils import utils
from chatlearn.utils.constant import LORA_WEIGHT_PREFIX
from chatlearn.utils.constant import PARAM_SYNC_COMM_TYPE
from chatlearn.utils.constant import ROUTED_EXPERT_REGROUPING_COMM_TYPE
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.utils.utils import execute_in_parallel
from chatlearn.utils.timer import Timers
from chatlearn.synchronizer.scheduler import CollectiveTask, parallel_execute_collective_tasks
from . import get_synchronizer

patch_ray()

class ParameterSyncGroup:
    """ParameterSyncGroup"""

    def __init__(self, src_model, dst_model, group_name, frequency, error_signal):
        self.src_model = src_model
        self.dst_model = dst_model
        self.synchronizer = get_synchronizer(src_model, dst_model)
        self.group_name = group_name
        self.error_signal = error_signal
        self.send_recv_actor_mappings = defaultdict(list)
        self.recv_send_actor_mappings = defaultdict(list)
        self.send_recv_actor_mappings_stage2 = defaultdict(list)
        self.recv_send_actor_mappings_stage2 = defaultdict(list)
        self.actor2rank = {}
        self.actor2model = {}
        self._debug = get_args().runtime_args.debug
        self._num_src_pipeline_stage = None
        self._num_dst_pipeline_stage = None
        self._num_src_expert_parallel = None
        self._num_dst_expert_parallel = None
        self._num_src_tensor_parallel = None
        self._num_dst_tensor_parallel = None
        self._send_recv_param_names = {}
        self._actor2pipe = {}
        self._actor2tp = {}
        self._actor2ep = {}
        self._actor2dp = {}
        self._comm_type = get_args().runtime_args.param_sync_comm_type
        if src_model.colocate_with(dst_model) and self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
            if self.num_src_tensor_parallel % 2 == 1 and self.num_dst_tensor_parallel % 2 == 1:
                logger.warning("Only support PARAM_SYNC_COMM_TYPE.BROADCAST when TP SIZE is even number, use P2P instead")
                self._comm_type = PARAM_SYNC_COMM_TYPE.P2P

        self.concurrent_comm = get_args().runtime_args.concurrent_comm
        self._enable_lora = self.src_model.module_args.lora.enable_lora
        # sync every n episodes, n = 0 for no param sync
        self._frequency = frequency

        self._free_sync_collective_group = get_args().runtime_args.free_sync_collective_group
        self._is_collective_group_created = True
        self.collective_groups = []
        self.groups2actors = {} # group_name -> []actors
        self.src_dp_size = future.get(self.src_model.replicas[0].all_actors[0].get_data_parallel_size.remote())
        self.send_actors_to_regroup_routed_experts = None
        self._comm_type_to_regroup_routed_experts = get_args().runtime_args.routed_expert_regrouping_comm_type
        assert self._comm_type_to_regroup_routed_experts in \
            [ROUTED_EXPERT_REGROUPING_COMM_TYPE.ALLGATHER, ROUTED_EXPERT_REGROUPING_COMM_TYPE.ALLTOALL], \
            f"Only support 'allgather' or 'alltoall' for routed expert regrouping, while {self._comm_type_to_regroup_routed_experts}"
        if self._comm_type_to_regroup_routed_experts == ROUTED_EXPERT_REGROUPING_COMM_TYPE.ALLTOALL:
            if self.num_dst_tensor_parallel * self.num_dst_expert_parallel != self.num_src_tensor_parallel * self.num_src_expert_parallel:
                logger.info("Only support ROUTED_EXPERT_REGROUPING_COMM_TYPE.ALLTOALL when src tp eqs dst tp, use 'allgather' instead.")
                self._comm_type_to_regroup_routed_experts = ROUTED_EXPERT_REGROUPING_COMM_TYPE.ALLGATHER
        logger.info(f"Set ROUTED_EXPERT_REGROUPING_COMM_TYPE = {self._comm_type_to_regroup_routed_experts}.")
        self.sorted_send_actors = None
        self.sorted_send_actors_stage2 = None
        self.actor2synchronizer = {}

        self.setup_collective_group()

        self.setup_rank_mapping()
        self.timers = Timers()

    def get_group_name(self, actors):
        return f"{self.group_name}_" + "_".join(str(self.actor2rank[actor]) for actor in actors)

    @property
    def frequency(self):
        return self._frequency

    def get_or_cache(self, actor, func_name, *args, **kwargs):
        def inner_func(*args, **kwargs):
            return future.get(getattr(getattr(actor, func_name), 'remote')(*args, **kwargs))
        cached_name = str(actor) + "_" + func_name
        if hasattr(self, cached_name):
            cached = getattr(self, cached_name)
        else:
            cached = {}
            setattr(self, cached_name, cached)
        return utils.get_or_cache(cached, actor, inner_func, *args, **kwargs)

    def is_same_gpu(self, src_actor, dst_actor):
        src_gpu = self.get_or_cache(src_actor, "get_visible_gpus")
        dst_gpu = self.get_or_cache(dst_actor, "get_visible_gpus")
        src_address = self.get_or_cache(src_actor, "get_address")
        dst_address = self.get_or_cache(dst_actor, "get_address")
        return src_gpu == dst_gpu and src_address == dst_address

    @property
    def num_src_pipeline_stage(self):
        if self._num_src_pipeline_stage is None:
            self._num_src_pipeline_stage = future.get(self.src_model.replicas[0].all_actors[0].pipeline_model_parallel_size.remote())
        return self._num_src_pipeline_stage

    @property
    def num_dst_pipeline_stage(self):
        if self._num_dst_pipeline_stage is None:
            self._num_dst_pipeline_stage = future.get(self.dst_model.replicas[0].all_actors[0].pipeline_model_parallel_size.remote())
        return self._num_dst_pipeline_stage

    @property
    def num_src_tensor_parallel(self):
        if self._num_src_tensor_parallel is None:
            self._num_src_tensor_parallel = future.get(self.src_model.replicas[0].all_actors[0].tensor_model_parallel_size.remote())
        return self._num_src_tensor_parallel

    @property
    def num_dst_tensor_parallel(self):
        if self._num_dst_tensor_parallel is None:
            self._num_dst_tensor_parallel = future.get(self.dst_model.replicas[0].all_actors[0].tensor_model_parallel_size.remote())
        return self._num_dst_tensor_parallel

    @property
    def num_src_expert_parallel(self):
        if self._num_src_expert_parallel is None:
            self._num_src_expert_parallel = future.get(self.src_model.replicas[0].all_actors[0].expert_model_parallel_size.remote())
        return self._num_src_expert_parallel

    @property
    def num_dst_expert_parallel(self):
        if self._num_dst_expert_parallel is None:
            self._num_dst_expert_parallel = future.get(self.dst_model.replicas[0].all_actors[0].expert_model_parallel_size.remote())
        return self._num_dst_expert_parallel

    def setup_collective_group(self):
        refs = []
        # we put src_model first, so we don't need to change the rank of training model
        models = [self.src_model, self.dst_model]
        world_size = sum(model.actor_num for model in models)

        rank_offset = 0
        for model in models:
            for replica in model.replicas:
                if self._comm_type == PARAM_SYNC_COMM_TYPE.P2P:
                    refs += replica._setup_collective_group(rank_offset, world_size, self.group_name)
                else:
                    replica._setup_ranks(rank_offset)
                rank_offset += replica.actor_num
        if refs:
            future.get(refs)
            logger.info(f"init collective group done for {self.group_name}")

    def destroy_collective_group(self):
        refs = []
        try:
            refs.extend(self.src_model.destroy_collective_group())
            refs.extend(self.dst_model.destroy_collective_group())
            future.wait(refs)
            logger.info(f"destroy_collective_group success for {self.group_name}")
        except Exception as e:
            logger.exception(f"destroy_collective_group fail for {self.group_name} {e}")

    def setup_rank_mapping(self):
        self.tp_num_mapping = self.num_dst_tensor_parallel // self.num_src_tensor_parallel
        if self.tp_num_mapping == 1:
            self.build_rank_mapping()
        else:
            self.build_rank_mapping_two_stage()

    def insert_actor2rank(self, actor, rank: int):
        if actor not in self.actor2rank:
            self.actor2rank[actor] = rank

    def insert_actor2model(self, actor, model):
        if actor not in self.actor2model:
            self.actor2model[actor] = model

    def add_routed_experts_regrouping_actor(self, model, ranks_group: List):
        for replica_ranks_group in ranks_group:
            if isinstance(replica_ranks_group[0], list):
                for tp_ranks in replica_ranks_group:
                    for rank in tp_ranks:
                        actor = model.get_actor(rank)
                        self.insert_actor2rank(actor, rank)
                        self.insert_actor2model(actor, model)
            else:
                for rank in replica_ranks_group:
                    actor = model.get_actor(rank)
                    self.insert_actor2rank(actor, rank)
                    self.insert_actor2model(actor, model)

    # pylint: disable=unused-argument
    def empty_add_recv_actor(self, src_rank, dst_rank):
        return

    def warmup_groups(self):
        return

    def add_recv_actor(self, src_rank, dst_rank):
        src_actor = self.src_model.get_actor(src_rank)
        self.insert_actor2rank(src_actor, src_rank)
        self.insert_actor2model(src_actor, self.src_model)
        dst_actor = self.dst_model.get_actor(dst_rank)
        self.insert_actor2rank(dst_actor, dst_rank)
        self.insert_actor2model(dst_actor, self.dst_model)

        src_gpu = self.get_or_cache(src_actor, "get_visible_gpus")
        dst_gpu = self.get_or_cache(dst_actor, "get_visible_gpus")
        src_tp_rank = self.get_actor_tp_rank(src_actor)
        dst_tp_rank = self.get_actor_tp_rank(dst_actor)
        src_pp_rank = self.get_actor_pipe_rank(src_actor)
        dst_pp_rank = self.get_actor_pipe_rank(dst_actor)
        src_ep_rank = self.get_actor_ep_rank(src_actor)
        dst_ep_rank = self.get_actor_ep_rank(dst_actor)
        logger.debug(f"build rank mapping from {src_rank} to {dst_rank}, from gpu {src_gpu} to {dst_gpu}, " +
                     f"from pipe_stage {src_pp_rank} to {dst_pp_rank}, " +
                     f"from tp rank {src_tp_rank} to {dst_tp_rank}, " +
                     f"from ep rank {src_ep_rank} to {dst_ep_rank}.")
        self.send_recv_actor_mappings[src_actor].append(dst_actor)
        self.recv_send_actor_mappings[dst_actor].append(src_actor)

    def add_recv_actor_stage2(self, src_rank, dst_rank):
        src_actor = self.dst_model.get_actor(src_rank)
        self.insert_actor2rank(src_actor, src_rank)
        self.insert_actor2model(src_actor, self.dst_model) # stage 2 sends from dst_model to dst_model
        dst_actor = self.dst_model.get_actor(dst_rank)
        self.insert_actor2rank(dst_actor, dst_rank)
        self.insert_actor2model(dst_actor, self.dst_model)

        src_gpu = future.get(src_actor.get_visible_gpus.remote())
        dst_gpu = future.get(dst_actor.get_visible_gpus.remote())
        # TODO(jiangle.jl): support ep/cp.
        src_tp_rank = self.get_actor_tp_rank(src_actor)
        dst_tp_rank = self.get_actor_tp_rank(dst_actor)
        src_pp_rank = self.get_actor_pipe_rank(src_actor)
        dst_pp_rank = self.get_actor_pipe_rank(dst_actor)
        logger.debug(f"build rank mapping from {src_rank} to {dst_rank}, from gpu {src_gpu} to {dst_gpu}, " + \
                     f"from pipe_stage {src_pp_rank} to {dst_pp_rank}, " + \
                     f"from tp rank {src_tp_rank} to {dst_tp_rank}")
        self.send_recv_actor_mappings_stage2[src_actor].append(dst_actor)
        self.recv_send_actor_mappings_stage2[dst_actor].append(src_actor)

    def set_send_actors_to_regroup_routed_experts(self, src_replica_ranks_group):
        if self.send_actors_to_regroup_routed_experts is None:
            self.send_actors_to_regroup_routed_experts = []
        for src_replica_ranks in src_replica_ranks_group:
            self.send_actors_to_regroup_routed_experts.append([])
            if isinstance(src_replica_ranks[0], list):
                for src_tp_ranks in src_replica_ranks:
                    self.send_actors_to_regroup_routed_experts[-1].extend(
                        [self.src_model.get_actor(src_rank) for src_rank in src_tp_ranks])
            else:
                self.send_actors_to_regroup_routed_experts[-1].extend(
                    [self.src_model.get_actor(src_rank) for src_rank in src_replica_ranks])

    def get_src_and_dst_dp_ranks(self, is_except_routed_experts=False):
        dst_dp_ranks = self.dst_model.all_ranks
        local_src_ranks = future.get(self.src_model.replicas[0].get_local_param_ranks())
        if local_src_ranks[0] is None or dst_dp_ranks is None:
            if self._debug:
                logger.warning(
                    f"DEBUG MODE! src_dp_ranks {local_src_ranks} or dst_dp_ranks: {dst_dp_ranks} is None, "
                    "make sure they have values in real application.")
                return local_src_ranks, dst_dp_ranks
            else:
                raise Exception(f"src_dp_ranks {local_src_ranks} or dst_dp_ranks {dst_dp_ranks} should not be None")
        dp_rank_to_ranks = defaultdict(list)
        for local_ranks, dp_rank in local_src_ranks:
            dp_rank_to_ranks[dp_rank].append(local_ranks[dp_rank])
        if is_except_routed_experts:
            # for weight except routed expert, ep_size using for data parallel.
            src_hep_size = self.num_src_expert_parallel * self.num_src_tensor_parallel
            new_dict = defaultdict(list)
            idx = 0
            for dp_rank, values in dp_rank_to_ranks.items():
                assert len(values) % src_hep_size == 0, (
                    f"len of values({len(values)}) for dp_rank {dp_rank} must be divisible by hep size({src_hep_size})"
                    f" when call get_src_and_dst_dp_ranks_for_except_routed_experts."
                )
                pp_blocks = [values[i:i + src_hep_size] for i in range(0, len(values), src_hep_size)]
                sub_blocks_per_pp = []
                for block in pp_blocks:
                    sub_block_size = src_hep_size // self.num_src_expert_parallel
                    sub_blocks = [block[i:i + sub_block_size] for i in range(0, src_hep_size, sub_block_size)]
                    sub_blocks_per_pp.append(sub_blocks)
                for i in range(self.num_src_expert_parallel):
                    merged_group = []
                    for sub_blocks in sub_blocks_per_pp:
                        merged_group.extend(sub_blocks[i])
                    new_dict[idx].extend(merged_group)
                    idx += 1
            src_dp_ranks = [i[1] for i in sorted(new_dict.items())]
        else:
            src_dp_ranks = [i[1] for i in sorted(dp_rank_to_ranks.items())]
        return src_dp_ranks, dst_dp_ranks

    def get_load_balance_dst_rank(
        self,
        lb_dst_offset_pq_dict,
        s_idx,
        start,
        src_rank,
        dst_replica_ranks_group,
        d_idx,
        pre_allocate=False
    ):
        """Get the dst_rank for load balance when gpu collides.
        """
        dst_tp_indices = sorted([
            s_idx * self.tp_num_mapping + (start + i) % self.tp_num_mapping
            for i in range(self.tp_num_mapping)
        ])
        indexed_dst_tp_group = tuple(dst_replica_ranks_group[d_idx][dst_tp_index] for dst_tp_index in dst_tp_indices)

        # Construct a priority queue (PQ) to retrieve `dst_rank` for load balancing when gpu collides.
        # The key of the PQ is (hit_time, max_seq_num), meaning that the rank is used for `hit_time` times,
        # while `max_seq_num` further sorts `dst_rank` when `hit_time` remains the same.
        if indexed_dst_tp_group not in lb_dst_offset_pq_dict:
            pq = PriorityQueue()
            max_seq_num = 0
            hit_time = 0
            while max_seq_num < self.tp_num_mapping:
                pq.put((
                    hit_time,
                    max_seq_num,
                    s_idx * self.tp_num_mapping + (start + max_seq_num) % self.tp_num_mapping
                ))
                max_seq_num += 1
            lb_dst_offset_pq_dict[indexed_dst_tp_group] = [pq, max_seq_num]
        else:
            max_seq_num = lb_dst_offset_pq_dict[indexed_dst_tp_group][1]

        # Each time, we retrieve the first value of the PQ.
        # 1. If the first `dst_rank` will encounter gpu collision with `src_rank`, we retrieve it, set `seq_num` to
        #    `max_seq_num`, increase `max_seq_num` by 1, and finally insert <(hit_time, seq_num), offset> back
        #    to the PQ.
        # 2. If the first `dst_rank` won't encounter gpu collision with `src_rank`, we insert it to another PQ (called
        #    legal_lb_recv_offset_pq). After looping through all legal solutions, we will retrieve the first load-balance one.
        # 3. If we cannot find a legal solution after `self.tp_num_mapping` times, all `src_rank` will encounter gpu collision
        #    with `dst_rank`, we throw a runtime exception.
        lb_recv_offset_pq = lb_dst_offset_pq_dict[indexed_dst_tp_group][0]
        legal_lb_recv_offset_pq = PriorityQueue()
        assert len(lb_recv_offset_pq.queue) == self.tp_num_mapping, (
            "length of the load-balance recv_offset priority queue must be equal to tp_num_mapping, "
            f"got {len(lb_recv_offset_pq.queue)} and {self.tp_num_mapping}."
        )
        is_collide = False
        for _ in range(self.tp_num_mapping):
            hit_time, seq_num, offset = lb_recv_offset_pq.get()
            dst_rank = dst_replica_ranks_group[d_idx][offset]
            logger.debug(f"Trying to match {src_rank} and {dst_rank} (hit={hit_time}), remaining queue={lb_recv_offset_pq.queue})")
            src_actor = self.src_model.get_actor(src_rank)
            dst_actor = self.dst_model.get_actor(dst_rank)
            if self.is_same_gpu(src_actor, dst_actor):
                logger.info(
                    f"src_rank ({src_rank}) will share the same gpu with dst_rank ({dst_rank}). "
                    "This is not allowed in NCCL send-recv. ChatLearn will skip dst_rank to the next legal one."
                )
                is_collide = True
                lb_recv_offset_pq.put((hit_time, max_seq_num, offset))
                max_seq_num += 1
                lb_dst_offset_pq_dict[indexed_dst_tp_group][1] = max_seq_num
            else:
                legal_lb_recv_offset_pq.put((hit_time, seq_num, offset))

        logger.debug(f"legal_lb_recv_offset_pq={legal_lb_recv_offset_pq.queue}")
        # if pre_allocate is True and no collide, we directly return
        if pre_allocate is True and is_collide is False:
            while len(legal_lb_recv_offset_pq.queue) > 0:
                lb_recv_offset_pq.put(legal_lb_recv_offset_pq.get())
            return None, False

        # there must be at least one legal recv offset
        if len(legal_lb_recv_offset_pq.queue) == 0:
            raise RuntimeError(
                f"Rank mapping solution is infeasible because src_rank ({src_rank}) will collide with all candidates."
            )

        # extract the first legal one to keep load balance
        hit_time, seq_num, offset = legal_lb_recv_offset_pq.get()
        lb_recv_offset_pq.put((hit_time + 1, seq_num, offset))

        # put other solutions back to lb_recv_offset_pq
        while len(legal_lb_recv_offset_pq.queue) > 0:
            lb_recv_offset_pq.put(legal_lb_recv_offset_pq.get())
        logger.debug(f"after retrieving, lb_recv_offset_pq = {lb_recv_offset_pq.queue}")

        # return dst_rank
        dst_rank = dst_replica_ranks_group[d_idx][offset]
        return dst_rank, is_collide

    def build_rank_mapping(self, add_recv_actor_fn=None):
        # setup rank mapping for src parameter and dst parameter
        # get rank for one src_model, without model replicas

        if add_recv_actor_fn is None:
            add_recv_actor_fn = self.add_recv_actor

        src_dp_ranks, dst_dp_ranks = self.get_src_and_dst_dp_ranks()
        if self._debug and (src_dp_ranks[0] is None or dst_dp_ranks is None):
            return

        if self.src_model.colocate_with(self.dst_model) and self.num_src_tensor_parallel % 2 == 1:
            replica_rank_iter = cycle(reversed(src_dp_ranks))
        else:
            replica_rank_iter = cycle(iter(src_dp_ranks))
        logger.debug(f"src_dp_ranks: {src_dp_ranks}")
        logger.debug(f"dst_dp_ranks: {dst_dp_ranks}")

        assert self.num_src_pipeline_stage % self.num_dst_pipeline_stage == 0

        def split_ranks_by_tp_and_ep_size(ranks,
                                          tp_size : int = 1,
                                          ep_size : int = 1):
            tp_and_ep_size = tp_size * ep_size
            return [ranks[i:i + tp_and_ep_size] for i in range(0, len(ranks), tp_and_ep_size)]

        for dst_replica_ranks in dst_dp_ranks:
            src_replica_ranks = next(replica_rank_iter)
            src_replica_ranks_group = split_ranks_by_tp_and_ep_size(src_replica_ranks, self.num_src_tensor_parallel, self.num_src_expert_parallel)
            dst_replica_ranks_group = split_ranks_by_tp_and_ep_size(dst_replica_ranks, self.num_dst_tensor_parallel, self.num_dst_expert_parallel)
            self.set_send_actors_to_regroup_routed_experts(src_replica_ranks_group)
            pipe_map_interval = self.num_src_pipeline_stage // self.num_dst_pipeline_stage
            for i, src_tp_group in enumerate(src_replica_ranks_group):
                j = i // pipe_map_interval
                for src_rank, dst_rank in zip(src_tp_group, dst_replica_ranks_group[j]):
                    add_recv_actor_fn(src_rank, dst_rank)

    # pylint: disable=unused-argument
    def build_rank_mapping_for_ep(self, add_recv_actor_fn=None):
        # Currently, we do not support build rank mapping for expert parallelism
        raise NotImplementedError("ChatLearn does not support build rank mapping from Megatron-LM for expert parallelism")

    def build_rank_mapping_two_stage(self, add_recv_actor_fn=None):
        # setup rank mapping for src parameter and dst parameter
        # get rank for one src_model, without model replicas

        if add_recv_actor_fn is None:
            add_recv_actor_stage1_fn = self.add_recv_actor
            add_recv_actor_stage2_fn = self.add_recv_actor_stage2
        else:
            assert len(add_recv_actor_fn) == 2, (
                "The length of add_recv_actor_fn should be 2. The first one is a function handler for communication stage 1, "
                "while the second one is a function handler for communication stage 2."
            )
            add_recv_actor_stage1_fn = add_recv_actor_fn[0]
            add_recv_actor_stage2_fn = add_recv_actor_fn[1]

        src_ranks, dst_ranks = self.get_src_and_dst_dp_ranks(is_except_routed_experts=True)
        if self._debug and (src_ranks[0] is None or dst_ranks is None):
            return

        replica_rank_iter = cycle(iter(src_ranks))

        logger.debug(f"src_ranks: {src_ranks}")
        logger.debug(f"dst_ranks: {dst_ranks}")
        assert self.num_dst_tensor_parallel % self.num_src_tensor_parallel == 0, \
            "currently we require mod value equals to zero for tensor_model_parallel_size of dst_model and that of src_model while " + \
            f"src model {self.src_model.name}(TP={self.num_src_tensor_parallel}) and " + \
            f"dst model {self.dst_model.name}(TP={self.num_dst_tensor_parallel})"
        assert self.num_src_pipeline_stage % self.num_dst_pipeline_stage == 0

        def split_ranks_by_tp_and_ep_size(ranks, tp_size, ep_size):
            if ep_size > 1:
                sort_ranks_on_grouped_tp = []
                index = 0
                tp_index = 0
                for _ in range(len(ranks)):
                    sort_ranks_on_grouped_tp.append(index)
                    if tp_index < tp_size - 1:
                        index += 1
                        tp_index += 1
                    else:
                        start_index = index + 1 - tp_size
                        index = start_index + (ep_size * tp_size)
                        tp_index = 0
                    if index >= len(ranks):
                        index = (index + tp_size) % len(ranks)
            else:
                sort_ranks_on_grouped_tp = ranks
            return [sort_ranks_on_grouped_tp[i:i + tp_size] for i in range(0, len(sort_ranks_on_grouped_tp), tp_size)]

        pair_list = []
        p2p_list = []
        src_replica_offset = 0
        lb_dst_offset_pq_dict = {}

        for dst_replica_ranks in dst_ranks:
            src_replica_ranks = next(replica_rank_iter)
            # for weight except routed expert, ep_size using for data parallel.
            src_replica_ranks_group = split_ranks_by_tp_and_ep_size(src_replica_ranks, self.num_src_tensor_parallel, 1)
            dst_replica_ranks_group = split_ranks_by_tp_and_ep_size(dst_replica_ranks, self.num_dst_tensor_parallel, self.num_dst_expert_parallel)
            logger.debug(f"src_replica_ranks_group: {src_replica_ranks_group}")
            logger.debug(f"dst_replica_ranks_group: {dst_replica_ranks_group}")
            pipe_map_interval = self.num_src_pipeline_stage // self.num_dst_pipeline_stage

            assert pipe_map_interval >= 1, \
                f"dst_pp expected to divide src_pp, while src_pp {self.num_src_pipeline_stage} and dst_pp {self.num_dst_pipeline_stage}"

            # stage 1: comm pairs that broadcast params from trainer to inference model
            # Each rank in trainer holds weights for tp_num_mapping ranks in inference model.
            # For example: trainer_tp = 2, inference_tp = 4 => tp_num_mapping = inference_tp // trainer_tp = 2
            # Weight mapping from training to inference:
            #   [0] -> [0', 1']
            #   [1] -> [2', 3']
            # To avoid p2p communication on the same gpu, we only broadcast params to first rank in weight_mapping_group.
            # Comm mapping from training to inference:
            #   [0] -> [0']
            #   [1] -> [2']
            # Firstly, pre-allocate for those gpu collisions
            uncollided_index_to_start_j = {}
            for i, src_tp_group in enumerate(src_replica_ranks_group):
                if i < src_replica_offset:
                    continue
                j = (i - src_replica_offset) // pipe_map_interval
                if j == self.num_dst_pipeline_stage:
                    src_replica_offset = i
                    break
                if self.tp_num_mapping == 1:
                    start =  0
                else:
                    mod_i = (i - src_replica_offset) % self.tp_num_mapping
                    start = mod_i if (i - src_replica_offset) < self.tp_num_mapping else (self.tp_num_mapping - mod_i - 1) % self.tp_num_mapping
                for s_idx, src_rank in enumerate(src_tp_group):
                    dst_rank, is_collide = self.get_load_balance_dst_rank(
                        lb_dst_offset_pq_dict,
                        s_idx,
                        start,
                        src_rank,
                        dst_replica_ranks_group,
                        j,
                        pre_allocate=True
                    )
                    if is_collide:
                        add_recv_actor_stage1_fn(src_rank, dst_rank)
                        pair_list.append((src_rank, dst_rank))
                    else:
                        assert dst_rank is None
                        uncollided_index_to_start_j.update({(i, s_idx) : (start, j)})

            # Then, allocate src_ranks without gpu collisions
            for i, src_tp_group in enumerate(src_replica_ranks_group):
                for s_idx, src_rank in enumerate(src_tp_group):
                    if (i, s_idx) not in uncollided_index_to_start_j:
                        continue

                    start, j = uncollided_index_to_start_j.get((i, s_idx))
                    dst_rank, _ = self.get_load_balance_dst_rank(
                        lb_dst_offset_pq_dict,
                        s_idx,
                        start,
                        src_rank,
                        dst_replica_ranks_group,
                        j,
                        pre_allocate=False
                    )
                    add_recv_actor_stage1_fn(src_rank, dst_rank)
                    pair_list.append((src_rank, dst_rank))

            # stage 2: comm pairs that broadcast params from first rank to the other ranks for each weight_mapping_group
            # Comm mapping in each weight_mapping_group of inference:
            #   [0'] -> [1']
            #   [2'] -> [3']
            recv_ranks = [pair[1] for pair in pair_list]
            def p2p_pair_grouping(tuples):
                for s_idx, src_rank in enumerate(tuples):
                    for d_idx, dst_rank in enumerate(tuples):
                        if s_idx == d_idx or src_rank not in recv_ranks: # pylint: disable=cell-var-from-loop
                            continue
                        add_recv_actor_stage2_fn(src_rank, dst_rank)
                        p2p_list.append((src_rank, dst_rank))

            for dst_tp_group in dst_replica_ranks_group:
                dst_tp_group = split_ranks_by_tp_and_ep_size(dst_tp_group, self.tp_num_mapping, 1)
                for tuples in dst_tp_group:
                    p2p_pair_grouping(tuples)

        logger.info(f"comm pair_list <train_rank, inference_rank>: {pair_list}")
        logger.info(f"comm p2p_list <inference_rank, inference_rank>: {p2p_list}")

    def _clear_sync_send_recv_parameters(self, rank_mappings:List):
        if len(rank_mappings) == 0:
            return
        refs = []
        flagged_actors = set()
        for rank_mapping in rank_mappings:
            if len(rank_mapping) == 0:
                continue
            for send_actor, recv_actors in rank_mapping.items():
                if send_actor not in flagged_actors:
                    refs.append(send_actor.clear_sync_send_recv_parameters.remote())
                    flagged_actors.add(send_actor)
                for recv_actor in recv_actors:
                    if recv_actor not in flagged_actors:
                        refs.append(recv_actor.clear_sync_send_recv_parameters.remote())
                        flagged_actors.add(recv_actor)
        future.get(refs)

    def _clear_send_recv_param_names(self):
        self._send_recv_param_names = {}

    def _clear_sorted_send_actors(self, sorted_send_actors_list:List):
        if len(sorted_send_actors_list) == 0:
            return
        for sorted_send_actors in sorted_send_actors_list:
            if sorted_send_actors is not None:
                sorted_send_actors = None

    def clear_cache(self, sorted_send_actors_list=None, rank_mapping_list=None):
        if sorted_send_actors_list is None:
            sorted_send_actors_list = [
                self.send_actors_to_regroup_routed_experts,
                self.sorted_send_actors,
                self.sorted_send_actors_stage2
            ]
        if rank_mapping_list is None:
            rank_mapping_list = [self.send_recv_actor_mappings, self.send_recv_actor_mappings_stage2]

        self._clear_sync_send_recv_parameters(rank_mapping_list)
        self._clear_send_recv_param_names()
        self._clear_sorted_send_actors(sorted_send_actors_list)

    def validate_sync_results(self, send_actor, recv_actors, requires_grad, filter_fn=None, param_group="default"):
        assert param_group in ("default", "routed", "except_routed"), (
            f"param_group must be one of 'default', 'routed', or 'except_routed', got {param_group}."
        )

        def validate():
            src_names, dst_names = self.set_sync_param_names(send_actor, recv_actors[0], requires_grad, filter_fn, param_group)
            # check the value of src model and tgt model
            pipe_stage = self.get_actor_pipe_rank(send_actor)
            res = [send_actor.reset_sync_parameters.remote(src_names, pipe_stage)]
            for recv_actor in recv_actors:
                res.append(recv_actor.reset_sync_parameters.remote(dst_names, pipe_stage))
            future.wait(res)

            src_names, dst_names = future.get([send_actor.get_parameter_to_sync_names.remote(pipe_stage),
                                               recv_actors[0].get_parameter_to_sync_names.remote(pipe_stage)])

            assert len(src_names) == len(dst_names), (
                f"expect the length of src_names and dst_names being the same, got {len(src_names)} and {len(dst_names)}"
            )

            # check the value of src model and tgt model
            names = list(zip(src_names, dst_names))
            for src_name, dst_name in tqdm(names):
                if param_group in ("default", "except_routed"):
                    src_tensor = future.get(send_actor.get_parameter_to_sync.remote(src_name, pipe_stage, True, self.tp_num_mapping > 1))
                elif param_group == "routed":
                    src_tensor = future.get(send_actor.get_parameter_to_sync.remote(src_name, pipe_stage, True))
                if src_tensor.isnan().any():
                    raise RuntimeError(f"weight {src_name} from send actor is nan, please check checkpoint or training process.")
                src_tensor_shape = src_tensor.shape
                for recv_actor in recv_actors:
                    dst_tensor = future.get(recv_actor.get_parameter_to_sync.remote(dst_name, pipe_stage, True))
                    if dst_tensor.isnan().any():
                        raise RuntimeError(f"weight {dst_name} in recv actor is nan, please check param sync.")
                    if param_group in ("default", "except_routed"):
                        if self.tp_num_mapping == 1:
                            # for trainer_tp == inference_tp
                            assert src_tensor.shape == dst_tensor.shape, (
                                f"after weight sync {src_name}: {src_tensor.shape} and {dst_name}: {dst_tensor.shape} do not match."
                            )
                            assert torch.allclose(src_tensor, dst_tensor, atol=1e-06), (
                                f"after weight sync {src_name}: {src_tensor} and {dst_name}: {dst_tensor} do not match."
                            )
                        else:
                            # for inference_tp % trainer_tp == 0 and inference_tp > trainer_tp
                            dst_tensor_shape = dst_tensor.shape
                            src_tensor = src_tensor.reshape(-1)
                            dst_tensor = dst_tensor.reshape(-1)
                            tp_slice = self.actor2rank[recv_actor] % self.tp_num_mapping
                            if src_tensor.shape == dst_tensor.shape:
                                src_tensor_slice = src_tensor
                            else:
                                assert (
                                    src_tensor.shape[0] % dst_tensor.shape[0] == 0 and
                                    src_tensor.shape[0] // dst_tensor.shape[0] == self.tp_num_mapping
                                ), (
                                    f"num of elements in src_tensor must be divided by that of dst_tensor. "
                                    f"while src {src_name}: {src_tensor_shape} and dst {dst_name}: {dst_tensor_shape}."
                                )
                                start = dst_tensor.shape[0] * tp_slice
                                end = start + dst_tensor.shape[0]
                                src_tensor_slice = src_tensor[start:end]
                            assert torch.allclose(src_tensor_slice, dst_tensor, atol=1e-06), (
                                f"after weight sync {src_name}_{tp_slice}: "
                                f"{src_tensor_slice.view(dst_tensor_shape)} and {dst_name}: {dst_tensor.view(dst_tensor_shape)} do not match."
                            )
                    elif param_group == "routed":
                        assert self.hep_num_mapping == 1
                        assert src_tensor.shape == dst_tensor.shape, (
                            f"after weight sync {src_name}: {src_tensor.shape} and {dst_name}: {dst_tensor.shape} do not match."
                        )
                        assert torch.allclose(src_tensor, dst_tensor, atol=1e-06), (
                            f"after weight sync {src_name}: {src_tensor} and {dst_name}: {dst_tensor} do not match."
                        )
            return True
        logger.info("Going to validate transmitted tensors...")
        validate()
        logger.info("Validation passed!")

    def set_sync_param_names_stage2(self, send_actor, recv_actor, to_rank, requires_grad, filter_fn=None, param_group="default"):
        send_names, _ = self.set_sync_param_names(send_actor, send_actor, requires_grad, filter_fn, param_group)
        refs = []
        refs.append(send_actor.set_send_parameters.remote(send_names, self.get_actor_pipe_rank(send_actor)))
        refs.append(recv_actor.set_recv_parameters.remote(to_rank, send_names, self.get_actor_pipe_rank(recv_actor)))
        future.get(refs)
        return send_names, send_names

    def sync_broadcast_two_stage(self, actors, group_name, requires_grad=None, stage2=False, filter_fn=None, param_group="default"):
        send_actor = actors[0]
        start_time = time.time()
        stage_str = "STAGE1" if stage2 is False else "STAGE2"
        for rank, recv_actor in enumerate(actors[1:]):
            if stage2:
                self.set_sync_param_names_stage2(send_actor, recv_actor, self.actor2rank[recv_actor], requires_grad, filter_fn, param_group)
            else:
                self.set_sync_param_names(send_actor, recv_actor, requires_grad, filter_fn, param_group)
                pipe_stage = self.get_actor_pipe_rank(send_actor)

                shape_refs = []
                shape_refs.append(send_actor.get_parameter_shape.remote(pipe_stage))
                shape_refs.append(recv_actor.get_parameter_shape.remote(pipe_stage))
                send_shape_list, recv_shape_list = future.get(shape_refs)

                buffer_num = {}
                tp_division = {}
                for send_name_and_shape, recv_name_and_shape in zip(send_shape_list, recv_shape_list):
                    send_param_num = send_name_and_shape[1].numel()
                    recv_param_num = recv_name_and_shape[1].numel()
                    # for group query attention, tensor might consist of tp part and dp part.
                    ele_buffer_num = 1 if send_param_num == recv_param_num else self.tp_num_mapping
                    buffer_num[recv_name_and_shape[0]] = ele_buffer_num
                    tp_division[send_name_and_shape[0]] = ele_buffer_num
                refs = []
                refs.append(recv_actor.set_tp_num_mapping.remote(self.tp_num_mapping))
                refs.append(recv_actor.set_buffer_num.remote(buffer_num))
                refs.append(send_actor.set_tp_num_mapping.remote(self.tp_num_mapping))
                refs.append(send_actor.set_tp_division.remote(tp_division))
                future.get(refs)
        refs = []
        pipe_stage = self.get_actor_pipe_rank(send_actor)
        send_rank = 0
        if stage2:
            assert len(actors) == 2, f"expect only 2 actors for stage2. \
                sync params of relative rank to other slices of inference model. while {len(actors)}"
        for rank, actor in enumerate(actors):
            sync_buffer_rank = self.actor2rank[actors[1]] if rank == 0 and stage2 else 0
            ref = actor.broadcast_parameter_two_stage.remote(
                self.actor2rank[actor], sync_buffer_rank, rank, send_rank, group_name, pipe_stage, stage2)
            refs.append(ref)
        rets = future.wait(refs, return_output=True)
        logger.info(f"sync_broadcast_two_stage done {stage_str} {group_name} using {time.time()-start_time} seconds")
        return rets

    def sync_broadcast(self, actors, group_name, requires_grad=None, filter_fn=None, param_group="default"):
        send_actor = actors[0]
        for recv_actor in actors[1:]:
            self.set_sync_param_names(send_actor, recv_actor, requires_grad, filter_fn, param_group)
        pipe_stage = self.get_actor_pipe_rank(send_actor)
        refs = []
        for rank, actor in enumerate(actors):
            ref = actor.broadcast_parameter.remote(rank, 0, group_name, pipe_stage)
            refs.append(ref)
        future.wait(refs, return_output=True)

    def sync_allgather(self, actors, group_name, requires_grad=None, filter_fn=None):
        # Currently, only routed experts are to be all-gathered.
        for actor in actors:
            self.set_sync_param_names(actor, actor, requires_grad, filter_fn, param_group="routed", should_map_name=False)
        pipe_stage = self.get_actor_pipe_rank(actors[0])
        refs = []
        for actor in actors:
            ref = actor.allgather_routed_expert_parameter.remote(group_name, pipe_stage)
            refs.append(ref)
        future.wait(refs, return_output=True)

    def sync_alltoall(self, actors, requires_grad=None, filter_fn=None):
        # Currently, only routed experts are to be synced with all-to-all.
        for actor in actors:
            self.set_sync_param_names(actor, actor, requires_grad, filter_fn, param_group="routed", should_map_name=False)
        pipe_stage = self.get_actor_pipe_rank(actors[0])
        refs = []
        logger.info(f"apply alltoall among {[self.actor2rank[actor] for actor in actors]}")
        for actor in actors:
            ref = actor.alltoall_routed_expert_parameter.remote(pipe_stage)
            refs.append(ref)
        future.wait(refs, return_output=True)

    def _sync_send_recv(self, send_actor, recv_actor, requires_grad=None, filter_fn=None, param_group="default"):
        self.set_sync_param_names(send_actor, recv_actor, requires_grad, filter_fn, param_group)
        pipe_stage = self.get_actor_pipe_rank(send_actor)
        is_the_same_gpu = self.is_same_gpu(send_actor, recv_actor)

        if is_the_same_gpu:
            name2ref = send_actor.ray_put_parameter.remote(self.group_name, pipe_stage)
            recv_ref = recv_actor.ray_get_parameter.remote(self.group_name, name2ref, pipe_stage)
            future.get(recv_ref)
        else:
            send_ref = send_actor.send_parameter.remote(self.actor2rank[recv_actor], self.group_name, pipe_stage)
            recv_ref = recv_actor.recv_parameter.remote(self.actor2rank[send_actor], self.group_name, pipe_stage)
            future.get([send_ref, recv_ref])
        logger.debug(f"sync all parameters from {send_actor} to {recv_actor}")

    def sync_send_recv(self, send_actor, recv_actor, requires_grad=None, filter_fn=None, param_group="default"):
        try:
            self._sync_send_recv(send_actor, recv_actor, requires_grad, filter_fn, param_group)
        except Exception:
            future.get(self.error_signal.set.remote(traceback.format_exc()))

    def check_param_names(self, send_actor, recv_actor, src_names, dst_names):
        ref0 = send_actor.check_param_exists.remote(src_names)
        ref1 = recv_actor.check_param_exists.remote(dst_names)
        states = future.get([ref0, ref1])
        if not states[0]:
            raise RuntimeError(f"Check src parameters to sync fail {src_names}")
        if not states[1]:
            raise RuntimeError(f"Check dst parameters to sync fail {dst_names}")

    def get_actor_pipe_rank(self, actor):
        def inner_func():
            return future.get(actor.pipeline_parallel_rank.remote())
        return utils.get_or_cache(self._actor2pipe, actor, inner_func)

    def get_actor_tp_rank(self, actor):
        def inner_func():
            return future.get(actor.tensor_parallel_rank.remote())
        return utils.get_or_cache(self._actor2tp, actor, inner_func)

    def get_actor_ep_rank(self, actor):
        def inner_func():
            return future.get(actor.expert_parallel_rank.remote())
        return utils.get_or_cache(self._actor2ep, actor, inner_func)

    def get_actor_dp_rank(self, actor):
        def inner_func():
            return future.get(actor.get_data_parallel_rank.remote())
        return utils.get_or_cache(self._actor2dp, actor, inner_func)

    def _set_sync_param_names(self, send_actor, recv_actor, requires_grad=None, filter_fn=None, param_group="default", should_map_name=True):
        if requires_grad is None:
            requires_grad = True
        if self._enable_lora:
            # TODO(jiangle.jl): support freeze layer.
            requires_grad = False
        assert param_group in ("default", "routed", "except_routed"), (
            f"param_group must be one of 'default', 'routed', or 'except_routed', got {param_group}."
        )

        if self.num_src_pipeline_stage > 1:
            dst_pipe_rank = self.get_actor_pipe_rank(recv_actor)
            dst_layer_offset = self.get_or_cache(recv_actor, "get_pipeline_stage_layer_offset")
            dst_src_mappings = future.get(send_actor.build_pipeline_layer_name_mapping.remote(
                                          self.num_dst_pipeline_stage, dst_pipe_rank, dst_layer_offset,
                                          requires_grad=requires_grad))
            dst_names = list(dst_src_mappings.keys())
            src_names = list(dst_src_mappings.values())
        else:
            src_names = dst_names = future.get(send_actor.get_parameter_names.remote(requires_grad=requires_grad))

        if self._enable_lora:
            src_names = [ele for ele in src_names if LORA_WEIGHT_PREFIX not in ele]
            dst_names = [ele for ele in dst_names if LORA_WEIGHT_PREFIX not in ele]

        if filter_fn is not None:
            src_names = filter_fn(src_names)
            dst_names = filter_fn(dst_names)

        synchronizer = get_synchronizer(self.src_model, self.dst_model)
        if should_map_name:
            src_names, dst_names = synchronizer.map_name_from_src_to_dst(send_actor, recv_actor, src_names, dst_names)
        else:
            # For routed experts which need to regroup expert first in trainer actors.
            synchronizer.map_name_from_src_to_dst(send_actor, recv_actor, src_names, dst_names)
        self.actor2synchronizer[send_actor] = synchronizer
        future.wait(send_actor.set_synchronizer.remote(synchronizer))

        self.check_param_names(send_actor, recv_actor, src_names, dst_names)
        dst_model = self.actor2model[recv_actor]
        if self.tp_num_mapping > 1 and ((not dst_model.use_vllm_backend and param_group != "routed") or dst_model.use_vllm_backend):
            key = (recv_actor, recv_actor, param_group)
            if key not in self._send_recv_param_names:
                self._send_recv_param_names[key] = (dst_names, dst_names)
            else:
                dst_names0 = self._send_recv_param_names[key][0]
                dst_names0 += dst_names
                self._send_recv_param_names[key] = (dst_names0, dst_names0)
        if not self.synchronizer.is_parameter_changed:
            pipe_stage = self.get_actor_pipe_rank(send_actor)
            refs = []
            refs.append(send_actor.set_sync_parameters.remote(src_names, pipe_stage))
            refs.append(recv_actor.set_sync_parameters.remote(dst_names, pipe_stage))
            future.get(refs)
        return src_names, dst_names

    def set_sync_param_names(self, send_actor, recv_actor, requires_grad=None, filter_fn=None, param_group="default", should_map_name=True):
        src_names, dst_names = utils.get_or_cache(self._send_recv_param_names, (send_actor, recv_actor, param_group), \
            lambda: self._set_sync_param_names(send_actor, recv_actor, requires_grad, filter_fn, param_group, should_map_name))
        logger.debug(f"{self.actor2rank[send_actor]} -> {self.actor2rank[recv_actor]}: {src_names[:5]} -> {dst_names[:5]}")
        pipe_stage = self.get_actor_pipe_rank(send_actor)

        refs = []
        refs.append(send_actor.reset_sync_parameters.remote(src_names, pipe_stage))
        refs.append(recv_actor.reset_sync_parameters.remote(dst_names, pipe_stage))
        future.get(refs)

        return src_names, dst_names

    def create_broadcast_group(self, send_actor, recv_actors, group_name=None, param_group="default"):
        actor_groups = [send_actor]
        actor_groups.extend(recv_actors)
        # Use self.actor2rank to ensure a globally unique number within a param_group.
        send_actor_rank = self.actor2rank[send_actor]
        recv_actor_ranks = '_'.join([str(self.actor2rank[actor]) for actor in recv_actors])
        # Always include self.group_name to ensure the name of a param_group is unique.
        if group_name is None:
            group_name = self.group_name
        elif not group_name.startswith(self.group_name + "_"):
            group_name = self.group_name + "_" + group_name
        finalized_group_name = f"{group_name}_{param_group}_from_{send_actor_rank}_to_{recv_actor_ranks}"
        logger.debug(f"finalized_group_name is {finalized_group_name}")
        logger.debug(f"current collevtive_groups is {self.collective_groups}")
        logger.debug(f"send_actor: {send_actor}, recv_actors: {recv_actors}")
        if finalized_group_name not in self.collective_groups:
            refs = []
            for rank, actor in enumerate(actor_groups):
                ref = actor.setup_collective_group.remote(rank, len(actor_groups), "nccl", finalized_group_name)
                refs.append(ref)
            future.wait(refs)
            self.collective_groups.append(finalized_group_name)
            self.groups2actors[finalized_group_name] = tuple(actor_groups)
        return actor_groups, finalized_group_name

    def create_allgather_group(self, actor_groups, group_name=None):
        # Use self.actor2rank to ensure a globally unique number within a param_group.
        actor_ranks = '_'.join([str(self.actor2rank[actor]) for actor in actor_groups])
        # Always include self.group_name to ensure the name of a param_group is unique.
        if group_name is None:
            group_name = self.group_name
        elif not group_name.startswith(self.group_name + "_"):
            group_name = self.group_name + "_" + group_name
        finalized_group_name = f"{group_name}_routed_among_{actor_ranks}"
        logger.debug(f"finalized_group_name is {finalized_group_name}")
        logger.debug(f"current collevtive_groups is {self.collective_groups}")
        if finalized_group_name not in self.collective_groups:
            refs = []
            for rank, actor in enumerate(actor_groups):
                ref = actor.setup_collective_group.remote(rank, len(actor_groups), "nccl", finalized_group_name)
                refs.append(ref)
            future.wait(refs)
            self.collective_groups.append(finalized_group_name)
        return actor_groups, finalized_group_name

    def sort_send_actors(self, send_recv_actor_mappings, sorted_send_actors):
        if sorted_send_actors is not None:
            return sorted_send_actors
        dp2send_actors = defaultdict(list)
        for send_actor in send_recv_actor_mappings:
            dp2send_actors[self.get_actor_dp_rank(send_actor)].append(send_actor)
        for dp_rank in dp2send_actors:
            send_actors = dp2send_actors[dp_rank]
            dp2send_actors[dp_rank] = sorted(send_actors, key=lambda x: self.actor2rank[x])
        sorted_send_actors = []
        dp_rank = 0
        while len(sorted_send_actors) < len(send_recv_actor_mappings):
            sorted_send_actors.append(dp2send_actors[dp_rank].pop(0))
            dp_rank += 1
            # dp_rank not in dp2send_actors happens when inference replica number less than training replica number
            if dp_rank == self.src_dp_size or dp_rank not in dp2send_actors:
                dp_rank = 0
        assert len(send_recv_actor_mappings) == len(sorted_send_actors)
        return sorted_send_actors

    def sync_broadcast_second_stage_internal(self, group_name, thread_group, requires_grad=None, filter_fn=None, param_group="default", dryrun=False):
        max_workers = len(thread_group)
        logger.info(f"Use {max_workers} workers for second_stage_internal broadcasting.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, actor_group in enumerate(thread_group):
                send_actor, recv_actor = actor_group
                group_name_with_idx = f"{group_name}_{idx}"
                actor_groups, finalized_group_name = self.create_broadcast_group(
                    send_actor, [recv_actor], group_name=group_name_with_idx, param_group=param_group
                )
                if dryrun:
                    continue
                futures.append(executor.submit(
                    self.sync_broadcast_two_stage, actor_groups, finalized_group_name, requires_grad, True, filter_fn, param_group))
            for _future in concurrent.futures.as_completed(futures):
                try:
                    _future.result()
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Parameter sync thread generated an exception: {e}") # pylint: disable=raise-missing-from
            concurrent.futures.wait(futures)


    def sync_broadcast_second_stage(self, group_name, thread_groups, requires_grad=None, filter_fn=None, param_group="default", dryrun=False):
        tp_size = self.num_dst_tensor_parallel
        num_thread_groups = len(thread_groups) // tp_size
        new_thread_groups = [thread_groups[tp_size*i:tp_size*(i+1)] for i in range(num_thread_groups)]

        if not new_thread_groups:
            new_thread_groups = [thread_groups]
        max_workers = 1

        logger.info(f"Use {max_workers} workers for second_stage broadcasting.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, thread_group in enumerate(new_thread_groups):
                group_name_with_idx = f"{group_name}_{idx}"
                futures.append(executor.submit(
                    self.sync_broadcast_second_stage_internal, group_name_with_idx, thread_group, requires_grad, filter_fn, param_group, dryrun))
            for _future in concurrent.futures.as_completed(futures):
                try:
                    _future.result()
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Parameter sync thread generated an exception: {e}") # pylint: disable=raise-missing-from
            concurrent.futures.wait(futures)

    def sync_broadcast_multi_threads(
        self, sorted_send_actors, send_recv_actor_mappings, max_workers=1, requires_grad=None,
        group_name=None, stage2=False, filter_fn=None, param_group="default", dryrun=False):

        if stage2:
            thread_group = []
            for send_actor in sorted_send_actors:
                recv_actors = send_recv_actor_mappings[send_actor]
                for recv_actor in recv_actors:
                    thread_group.append((send_actor, recv_actor))
            actor_groups_to_sync = []
            for group in thread_group:
                new_actor_group_flag = True
                for idx, actor_groups in enumerate(actor_groups_to_sync):
                    in_actor_group = False
                    for actor_group in actor_groups:
                        if group[0] in actor_group or group[1] in actor_group:
                            in_actor_group = True
                    if not in_actor_group:
                        new_actor_group_flag = False
                        actor_groups_to_sync[idx].append(group) #pylint: disable=unnecessary-list-index-lookup
                        break
                if new_actor_group_flag or not actor_groups_to_sync:
                    actor_groups_to_sync.append([group])

            for group_idx, actor_groups in enumerate(actor_groups_to_sync):
                if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                    self.sync_broadcast_second_stage(
                        f"{group_name}_{group_idx}",
                        actor_groups,
                        requires_grad,
                        filter_fn,
                        param_group,
                        dryrun=dryrun
                    )
                else:
                    raise RuntimeError("support p2p only for scenes that trainer_tp not equal to inference_tp.")
        else:
            max_workers = len(sorted_send_actors)
            logger.info(f"Use {max_workers} workers for first_stage broadcasting.")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for send_actor in sorted_send_actors:
                    recv_actors = send_recv_actor_mappings[send_actor]
                    if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                        actor_groups, finalized_group_name = self.create_broadcast_group(
                            send_actor, recv_actors, group_name=group_name, param_group=param_group
                        )
                        if not dryrun:
                            futures.append(executor.submit(
                                self.sync_broadcast_two_stage, actor_groups, finalized_group_name, requires_grad, stage2, filter_fn, param_group
                            ))
                    else:
                        raise RuntimeError("support p2p only for scenes that trainer_tp not equal to inference_tp.")
                for _future in concurrent.futures.as_completed(futures):
                    try:
                        _future.result()
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Parameter sync thread generated an exception: {e}") # pylint: disable=raise-missing-from
                concurrent.futures.wait(futures)

    def sync_allgather_multi_threads(
        self, send_actors, max_workers=1, requires_grad=None,
        group_name=None, filter_fn=None
    ):
        send_actors_to_allgather_routed_experts = send_actors[0]
        logger.info(f"Use {max_workers} workers for allgather multiprocessing.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for allgather_actors in send_actors_to_allgather_routed_experts:
                actor_groups, finalized_group_name = self.create_allgather_group(allgather_actors, group_name=group_name)
                futures.append(executor.submit(
                    self.sync_allgather, actor_groups, finalized_group_name, requires_grad, filter_fn=filter_fn
                ))
            for _future in concurrent.futures.as_completed(futures):
                try:
                    _future.result()
                except Exception as e:
                    raise RuntimeError(f"Parameter sync thread generated an exception: {e}") # pylint: disable=raise-missing-from
            concurrent.futures.wait(futures)

    def sync_alltoall_multi_threads(
        self, send_actors, max_workers=1, requires_grad=None, filter_fn=None
    ):
        send_actors_to_alltoall_routed_experts = send_actors[0]
        max_workers = len(send_actors_to_alltoall_routed_experts)
        logger.info(f"Use {max_workers} workers for alltoall multiprocessing.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for actor_groups in send_actors_to_alltoall_routed_experts:
                futures.append(executor.submit(
                    self.sync_alltoall, actor_groups, requires_grad, filter_fn=filter_fn
                ))
            for _future in concurrent.futures.as_completed(futures):
                try:
                    _future.result()
                except Exception as e:
                    raise RuntimeError(f"Parameter sync thread generated an exception: {e}") # pylint: disable=raise-missing-from
            concurrent.futures.wait(futures)

    def check_and_setup_collective_group(self):
        if not self._is_collective_group_created:
            # Re-create collective group only when it is destroyed before.
            assert self._free_sync_collective_group
            self.setup_collective_group()

    def check_and_destroy_collective_group(self):
        if self._free_sync_collective_group:
            self.destroy_collective_group()
            self._is_collective_group_created = False
            self.collective_groups = []
            self.groups2actors = {}

    def check_and_fuse_lora(self, enable_lora, actor_mapping):
        send_actors_set = set()

        def check_and_fuse_lora_internal(actor_mapping_item):
            for send_actor in actor_mapping_item:
                if enable_lora and send_actor not in send_actors_set:
                    ref = send_actor.fuse_lora_layer.remote()
                    state = future.get([ref])
                    assert state, "Check fuse lora layer fail."
                    send_actors_set.add(send_actor)

        if isinstance(actor_mapping, List):
            for actor_mapping_item in actor_mapping:
                if actor_mapping_item is None:
                    continue
                check_and_fuse_lora_internal(actor_mapping_item)
        elif isinstance(actor_mapping, Dict):
            if actor_mapping is None:
                return
            check_and_fuse_lora_internal(actor_mapping)
        else:
            raise ValueError("unrecognized type for actor_mapping, expect: List or Dict")

    def check_and_unfuse_lora(self, enable_lora, actor_mapping):
        send_actors_set = set()

        def check_and_unfuse_lora_internal(actor_mapping_item):
            for send_actor in actor_mapping_item:
                if self._enable_lora and send_actor not in send_actors_set:
                    ref = send_actor.unfuse_lora_layer.remote()
                    state = future.get([ref])
                    assert state, "Check unfuse lora layer fail."
                    send_actors_set.add(send_actor)

        if isinstance(actor_mapping, List):
            for actor_mapping_item in actor_mapping:
                if actor_mapping_item is None:
                    continue
                check_and_unfuse_lora_internal(actor_mapping_item)
        elif isinstance(actor_mapping, Dict):
            if actor_mapping is None:
                return
            check_and_unfuse_lora_internal(actor_mapping)
        else:
            raise ValueError("unrecognized type for actor_mapping, expect: List or Dict")

    def validate_sync_results_parallel(self, actor_mappings_list:List, requires_grad=None, validate=False, filter_fn=None, param_group="default"):
        if self._debug or validate:
            assert len(actor_mappings_list) in (1, 2), f"The length of actor mapping list should be 1 or 2, but got {len(actor_mappings_list)}."
            args = []
            for send_actor, recv_actors in actor_mappings_list[0].items():
                for recv_actor in recv_actors:
                    if len(actor_mappings_list) == 1:
                        args.append((send_actor, [recv_actor], requires_grad, filter_fn, param_group))
                    elif len(actor_mappings_list) == 2:
                        recv_actors_stage2 = actor_mappings_list[1].get(recv_actor, [])
                        args.append((send_actor, [recv_actor] + recv_actors_stage2, requires_grad, filter_fn, param_group))
            if self._debug:
                for arg in args:
                    self.validate_sync_results(arg[0], arg[1], arg[2], arg[3], arg[4])
            else:
                execute_in_parallel(self.validate_sync_results, args)

    def _calculate_max_workers(self, sorted_send_actors, actor_mappings=None):
        max_workers = get_args().runtime_args.param_sync_max_workers
        if max_workers is None:
            max_workers = max(self.src_model.total_gpu // self.num_src_pipeline_stage, 1)
        if max_workers == -1:
            if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                max_workers = len(sorted_send_actors)
            else:
                assert actor_mappings is not None, (
                    "actor_mappings should not be None when max_workers is -1 and "
                    "communication type for parameter synchronization is not broadcast."
                )
                max_workers = len(sorted_send_actors) * len(actor_mappings[sorted_send_actors[0]])
        return max_workers

    def _multi_thread_sync_for_tp_num_mapping_gt_1(
        self,
        send_actors:List,
        actor_mappings:List,
        requires_grad=None,
        filter_fn=None,
        param_group="default",
        dryrun=False
    ):
        assert len(send_actors) == 2, (
            f"Expect the length of send_actors being 2 for TP num mapping greater than 1, but got {len(send_actors)}."
        )
        send_actors_stage1 = send_actors[0] # pylint: disable=unused-variable
        send_actors_stage2 = send_actors[1] # pylint: disable=unused-variable

        assert len(actor_mappings) == 2, (
            f"Expect the length of actor_mappings being 2 for TP num mapping greater than 1, but got {len(actor_mappings)}."
        )
        actor_mappings_stage1 = actor_mappings[0]
        actor_mappings_stage2 = actor_mappings[1]

        # stage 1
        self.timers("stage1").start()

        sorted_send_actors_stage1 = list(actor_mappings_stage1.keys())
        max_workers = self._calculate_max_workers(sorted_send_actors_stage1, actor_mappings_stage1)
        group_name = self.group_name + "_stage1_comm"
        self.sync_broadcast_multi_threads(
            sorted_send_actors_stage1, actor_mappings_stage1, max_workers, requires_grad,
            group_name=group_name, stage2=False, filter_fn=filter_fn, param_group=param_group,
            dryrun=dryrun
        )
        self.timers("stage1").stop()
        logger.info(f"finish stage1| {self.timers.log(names=['stage1'])}")
        # stage 2
        self.timers("stage2").start()
        sorted_send_actors_stage2 = list(actor_mappings_stage2.keys())
        max_workers = self._calculate_max_workers(sorted_send_actors_stage2, actor_mappings_stage2)
        group_name = self.group_name + "_stage2_comm"
        self.sync_broadcast_multi_threads(
            sorted_send_actors_stage2, actor_mappings_stage2, max_workers, requires_grad,
            group_name=group_name, stage2=True, filter_fn=filter_fn, param_group=param_group,
            dryrun=dryrun)
        self.timers("stage2").stop()
        logger.info(f"finish stage2| {self.timers.log(names=['stage2'])}")

    def split_sync_groups(self, send_actors, actor_mappings):
        groups = []
        for send_actor in send_actors:
            recv_actors = actor_mappings[send_actor]
            rank_dict = [self.actor2rank[actor] for actor in [send_actor] + recv_actors]
            # gen groups
            placed = False
            for group in groups:
                if set(group["values"]).isdisjoint(rank_dict):
                    group["keys"].append(send_actor)
                    group["values"] = group["values"] + rank_dict
                    placed = True
                    break
            if not placed:
                groups.append({
                    "keys": [send_actor],
                    "values": rank_dict.copy()
                })
        total_elements = sum(len(group["keys"]) for group in groups)
        assert total_elements == len(send_actors), \
                (f"needed total elements of groups {total_elements} == len of send_actors \
                {len(send_actors)} in param sync.")
        for group in groups:
            assert len(group["values"]) == len(set(group["values"])), \
                (f"the elements must be all different in group: {group['values']}")
        logger.info(f"split_sync_groups: {groups}")
        return [g["keys"] for g in groups]

    def _multi_thread_sync_for_tp_num_mapping_eq_1(
        self, send_actors_list:List, actor_mappings_list:List,
        requires_grad=None, filter_fn=None, param_group="default", dryrun=False
    ):
        assert len(send_actors_list) == 1 and len(actor_mappings_list) == 1
        send_actors = send_actors_list[0]
        actor_mappings = actor_mappings_list[0]

        sorted_send_actors = self.sort_send_actors(actor_mappings, send_actors)
        max_workers = self._calculate_max_workers(sorted_send_actors, actor_mappings)
        src_pp_size = self.num_src_pipeline_stage

        groups = self.split_sync_groups(sorted_send_actors, actor_mappings)
        logger.info(f"Use {max_workers} workers for tp_num_mapping_eq_1 synchoronization, \
                src_pp_size: {src_pp_size}, groups: {len(groups)}.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            for group in groups:
                t1 = time.time()
                futures = []
                for send_actor in group:
                    recv_actors = actor_mappings[send_actor]
                    logger.info(f"Sending from {[self.actor2rank[send_actor]]} to {[self.actor2rank[actor] for actor in recv_actors]}.")
                    if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                        actor_groups, finalized_group_name = self.create_broadcast_group(send_actor, recv_actors, param_group=param_group)
                        if dryrun:
                            continue
                        futures.append(executor.submit(
                            self.sync_broadcast, actor_groups, finalized_group_name, requires_grad, filter_fn=filter_fn, param_group=param_group
                        ))
                    else:
                        for recv_actor in recv_actors:
                            if dryrun:
                                continue
                            futures.append(executor.submit(
                                self.sync_send_recv, send_actor, recv_actor, requires_grad, filter_fn=filter_fn, param_group=param_group
                            ))

                t2 = time.time()
                for _future in concurrent.futures.as_completed(futures):
                    try:
                        _future.result()
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Parameter sync thread generated an exception: {e}") from e
                concurrent.futures.wait(futures)
                t3 = time.time()
                logger.info(f"sync for tp_num_mapping_eq_1, submit time(s):{(t2-t1)}, sync time(s):{(t3-t2)}")

    def _single_thread_sync(self, actor_mappings_list:List, requires_grad=None, filter_fn=None, param_group="default"):
        assert len(actor_mappings_list) == 1
        actor_mappings = actor_mappings_list[0]

        for send_actor, recv_actors in actor_mappings.items():
            if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                actor_groups, finalized_group_name = self.create_broadcast_group(send_actor, recv_actors, param_group=param_group)
                self.sync_broadcast(actor_groups, finalized_group_name, requires_grad, filter_fn=filter_fn, param_group=param_group)
            else:
                for recv_actor in recv_actors:
                    self.sync_send_recv(send_actor, recv_actor, requires_grad, filter_fn=filter_fn, param_group=param_group)

    def recover_synchronizer(self):
        refs = []
        for actor, synchronizer in self.actor2synchronizer.items():
            refs.append(actor.set_synchronizer.remote(synchronizer))
        future.wait(refs)

    def reset_synchronizer(self):
        refs = []
        for actor, _ in self.actor2synchronizer.items():
            refs.append(actor.set_synchronizer.remote(None))
        future.wait(refs)

    def sync(self, requires_grad=None, validate=False, dryrun=False):
        self.recover_synchronizer()

        self.check_and_setup_collective_group()

        self.check_and_fuse_lora(self._enable_lora, self.send_recv_actor_mappings)

        send_actors_list : List = []
        actor_mappings_list : List = []
        if self.concurrent_comm:
            if self.tp_num_mapping > 1:
                send_actors_list = [self.sorted_send_actors, self.sorted_send_actors_stage2]
                actor_mappings_list = [self.send_recv_actor_mappings, self.send_recv_actor_mappings_stage2]
                self._multi_thread_sync_for_tp_num_mapping_gt_1(
                    send_actors_list,
                    actor_mappings_list,
                    requires_grad=requires_grad
                )
            else:
                send_actors_list = [self.sorted_send_actors]
                actor_mappings_list = [self.send_recv_actor_mappings]
                self._multi_thread_sync_for_tp_num_mapping_eq_1(
                    send_actors_list,
                    actor_mappings_list,
                    requires_grad=requires_grad
                )
        else:
            actor_mappings_list = [self.send_recv_actor_mappings]
            self._single_thread_sync(
                actor_mappings_list,
                requires_grad=requires_grad
            )

        assert len(actor_mappings_list) >= 1

        self.check_and_unfuse_lora(self._enable_lora, self.send_recv_actor_mappings)

        self.validate_sync_results_parallel(actor_mappings_list, requires_grad, validate)

        self.check_and_destroy_collective_group()

        self.reset_synchronizer()

        logger.info(f"Group {self.group_name} sync all parameters done, comm_type {self._comm_type}")


class ParameterSyncGroupwithHEP(ParameterSyncGroup):
    """ParameterSyncGroup for Hyper Expert Parallel (HEP).
    
       Note that in HEP, EP size for routed experts is different from that for Megatorn-LM. For routed experts,
       the new EP size (we call it HEP size for clarification) = mpu.ep_size x mpu.tp_size, while Megatron-LM
       set the EP size as mpu.ep_size. However, the EP size of shared experts in HEP is equal to that in Megatron
       -LM (which is 1). In this case, routed experts treat TP and EP altogether as EP, and shared experts ignore
       EP just like other non-expert weights. Therefore, we manage seperate parameter sync groups for routed
       expert weigts and weights except routed experts in this class.
    """

    def __init__(self, src_model, dst_model, group_name, frequency, error_signal):
        self.send_recv_actor_mappings_for_routed_experts = defaultdict(list)
        self.recv_send_actor_mappings_for_routed_experts = defaultdict(list)
        self._num_src_hyper_expert_parallel = None
        self._num_dst_hyper_expert_parallel = None
        self._actor2hep = {}
        self.sorted_send_actors_for_routed_experts = None
        super().__init__(src_model, dst_model, group_name, frequency, error_signal)

    def setup_rank_mapping(self):
        self.tp_num_mapping = self.num_dst_tensor_parallel // self.num_src_tensor_parallel
        self.ep_num_mapping = self.num_dst_expert_parallel / self.num_src_expert_parallel
        self.hep_num_mapping = self.num_dst_hyper_expert_parallel / self.num_src_hyper_expert_parallel
        assert self.tp_num_mapping >= 1, (
            f"Currently, tensor parallel world size for training ({self.num_src_tensor_parallel}) should be"
            f"less or equal to tensor parallel world size for inference ({self.num_dst_tensor_parallel}) with HEP enabled."
        )
        assert self.ep_num_mapping <= 1, (
            f"Currently, expert parallel world size for training ({self.num_src_expert_parallel}) should be"
            f"greater or equal to expert parallel world size for inference ({self.num_dst_expert_parallel}) with HEP enabled."
        )
        if self.dst_model.use_vllm_backend:
            if self.tp_num_mapping == 1:
                if self.ep_num_mapping == 1:
                    self.build_rank_mapping()
                else:
                    self.build_rank_mapping_for_ep()
            elif self.tp_num_mapping > 1:
                if self.hep_num_mapping == 1:
                    self.build_rank_mapping_for_ep(add_recv_actor_fn=self.add_recv_actor_for_routed_experts) # only add all-gather actors
                    self.build_rank_mapping_for_params_except_routed_expert()
                else:
                    self.build_rank_mapping_for_ep(add_recv_actor_fn=self.empty_add_recv_actor) # only add all-gather actors
                    self.build_rank_mapping_two_stage()
            else:
                raise NotImplementedError(
                    f"ChatLearn does not support synchronizing from larger tp size ({self.num_src_tensor_parallel})"
                    f"to smaller tp size ({self.num_dst_tensor_parallel}) currently."
                )
        else:
            if self.ep_num_mapping == 1 and self.tp_num_mapping == 1:
                self.build_rank_mapping()
            elif self.hep_num_mapping == 1:
                # In this case, routed experts are mapped one by one, while params except routed experts are split by TP.
                self.build_rank_mapping_for_routed_experts()
                self.build_rank_mapping_for_params_except_routed_expert()
            else:
                # We do not support other cases for HEP. Please note that tp_num_mapping > 1 with ep_num_mapping = 1 is also unsupported.
                raise NotImplementedError(
                    "ChatLearn does not support inequivalent EP x TP between training and inference with Hyper Expert Parallel (HEP) enabled and "
                    f"inference model is an instance of `MegatronModule`. Your current setting is "
                    f"EP{self.num_src_expert_parallel} TP{self.num_src_tensor_parallel} for training model `{self.src_model.name}` "
                    f"and EP{self.num_dst_expert_parallel} TP{self.num_dst_tensor_parallel} for inference model `{self.dst_model.name}`."
                )

    def build_rank_mapping_for_ep(self, add_recv_actor_fn=None):
        # setup rank mapping for src parameter and dst parameter
        # get rank for one src_model, without model replicas

        if add_recv_actor_fn is None:
            add_recv_actor_fn = self.add_recv_actor

        src_dp_ranks, dst_dp_ranks = self.get_src_and_dst_dp_ranks()
        if self._debug and (src_dp_ranks[0] is None or dst_dp_ranks is None):
            return

        assert len(src_dp_ranks[0]) % len(dst_dp_ranks[0]) == 0, \
            f"src training model ranks should be times of dst ranks, but got {len(src_dp_ranks[0])} and {len(dst_dp_ranks[0])}"
        if self.src_model.colocate_with(self.dst_model) and self.num_src_tensor_parallel % 2 == 1:
            replica_rank_iter = cycle(reversed(src_dp_ranks))
        else:
            replica_rank_iter = cycle(iter(src_dp_ranks))
        logger.debug(f"src_dp_ranks: {src_dp_ranks}")
        logger.debug(f"dst_dp_ranks: {dst_dp_ranks}")

        assert self.num_src_pipeline_stage % self.num_dst_pipeline_stage == 0

        def split_ranks_by_ep_and_tp_size(ranks,
                                          tp_size : int = 1,
                                          ep_size : int = 1):
            tp_and_ep_size = tp_size * ep_size
            return [[ranks[i:i + tp_size] for i in range(j, j + tp_and_ep_size, tp_size)] for j in range(0, len(ranks), tp_and_ep_size)]

        src_replica_ranks2offset = {}
        is_first_time_set_send_actors = True
        for dst_replica_ranks in dst_dp_ranks:
            src_replica_ranks = next(replica_rank_iter)
            if tuple(src_replica_ranks) not in src_replica_ranks2offset:
                src_replica_ranks2offset[tuple(src_replica_ranks)] = 0
                is_first_time_set_send_actors = True
            else:
                is_first_time_set_send_actors = False

            src_replica_ranks_group = split_ranks_by_ep_and_tp_size(src_replica_ranks, self.num_src_tensor_parallel, self.num_src_expert_parallel)
            # Since dst replica is vllm and it doesn't have ep, the function will organize dst_replica_ranks_group as [pp[tp]] naturally.
            dst_replica_ranks_group = split_ranks_by_ep_and_tp_size(dst_replica_ranks, self.num_dst_tensor_parallel, self.num_dst_expert_parallel)

            if is_first_time_set_send_actors:
                self.set_send_actors_to_regroup_routed_experts(src_replica_ranks_group)
                self.add_routed_experts_regrouping_actor(self.src_model, src_replica_ranks_group)

            if add_recv_actor_fn is self.empty_add_recv_actor:
                continue

            pipe_map_interval = self.num_src_pipeline_stage // self.num_dst_pipeline_stage
            for i, src_ep_and_tp_group in enumerate(src_replica_ranks_group):
                j = i // pipe_map_interval
                first_src_tp_group = src_ep_and_tp_group[0]
                assert len(dst_replica_ranks_group[j][0]) % len(first_src_tp_group) == 0, (
                    "TP size of dst model should be times of src model, "
                    f"but got {len(dst_replica_ranks_group[j][0])} and {len(first_src_tp_group)}"
                )
                len_dst_div_src = len(dst_replica_ranks_group[j][0]) // len(first_src_tp_group)
                concated_src_tp_group = []
                offset = src_replica_ranks2offset[tuple(src_replica_ranks)]
                # cycled concatenate src tp group to ensure len(concat_src_tp_group) == len(dst_replica_ranks_group[j][0])
                for k in range(len_dst_div_src):
                    concated_src_tp_group.extend(src_ep_and_tp_group[int((offset + k) % len(src_ep_and_tp_group))])
                for src_rank, dst_rank in zip(concated_src_tp_group, dst_replica_ranks_group[j][0]):
                    add_recv_actor_fn(src_rank, dst_rank)
                src_replica_ranks2offset[tuple(src_replica_ranks)] = int(
                    (src_replica_ranks2offset[tuple(src_replica_ranks)] + len_dst_div_src) % len(src_ep_and_tp_group)
                )

        if self._debug:
            def debug_msg_for_actor_mappings(actor_mapping):
                if actor_mapping is None:
                    return

                for k, v_list in actor_mapping.items():
                    for v in v_list:
                        logger.debug(f"actor_mappings: {self.actor2rank[k]} -> {self.actor2rank[v]}")

            debug_msg_for_actor_mappings(self.send_recv_actor_mappings)
            debug_msg_for_actor_mappings(self.send_recv_actor_mappings_for_routed_experts)

            for regroup_actors in self.send_actors_to_regroup_routed_experts:
                count += 1
                cat_str = "_".join(str(self.actor2rank[actor]) for actor in regroup_actors)
                logger.info(f"{self._comm_type_to_regroup_routed_experts} actors: {cat_str}")
            for k, v_list in self.send_recv_actor_mappings.items():
                for v in v_list:
                    logger.info(f"send_recv_actor_mappings: {self.actor2rank[k]} -> {self.actor2rank[v]}")

    def add_recv_actor_for_routed_experts(self, src_rank, dst_rank):
        src_actor = self.src_model.get_actor(src_rank)
        self.insert_actor2rank(src_actor, src_rank)
        self.insert_actor2model(src_actor, self.src_model)
        dst_actor = self.dst_model.get_actor(dst_rank)
        self.insert_actor2rank(dst_actor, dst_rank)
        self.insert_actor2model(dst_actor, self.dst_model)

        src_gpu = self.get_or_cache(src_actor, "get_visible_gpus")
        dst_gpu = self.get_or_cache(dst_actor, "get_visible_gpus")
        src_tp_rank = self.get_actor_tp_rank(src_actor)
        dst_tp_rank = self.get_actor_tp_rank(dst_actor)
        src_pp_rank = self.get_actor_pipe_rank(src_actor)
        dst_pp_rank = self.get_actor_pipe_rank(dst_actor)
        src_ep_rank = self.get_actor_ep_rank(src_actor)
        dst_ep_rank = self.get_actor_ep_rank(dst_actor)
        src_hep_rank = self.get_actor_hep_rank(src_actor)
        dst_hep_rank = self.get_actor_hep_rank(dst_actor)
        logger.debug(f"build rank mapping from {src_rank} to {dst_rank}, from gpu {src_gpu} to {dst_gpu}, " +
                     f"from pipe_stage {src_pp_rank} to {dst_pp_rank}, " +
                     f"from tp rank {src_tp_rank} to {dst_tp_rank}, " +
                     f"from ep rank {src_ep_rank} to {dst_ep_rank}, " + 
                     f"from hep rank {src_hep_rank} to {dst_hep_rank}.")
        self.send_recv_actor_mappings_for_routed_experts[src_actor].append(dst_actor)
        self.recv_send_actor_mappings_for_routed_experts[dst_actor].append(src_actor)

    @property
    def num_src_hyper_expert_parallel(self):
        if self._num_src_hyper_expert_parallel is None:
            self._num_src_hyper_expert_parallel = future.get(self.src_model.replicas[0].all_actors[0].tensor_and_expert_model_parallel_size.remote())
        return self._num_src_hyper_expert_parallel

    @property
    def num_dst_hyper_expert_parallel(self):
        if self._num_dst_hyper_expert_parallel is None:
            self._num_dst_hyper_expert_parallel = future.get(self.dst_model.replicas[0].all_actors[0].tensor_and_expert_model_parallel_size.remote())
        return self._num_dst_hyper_expert_parallel

    def get_actor_hep_rank(self, actor):
        def inner_func():
            return future.get(actor.tensor_and_expert_model_parallel_size.remote())
        return utils.get_or_cache(self._actor2hep, actor, inner_func)

    def build_rank_mapping_for_routed_experts(self):
        self.build_rank_mapping(add_recv_actor_fn=self.add_recv_actor_for_routed_experts)

    def build_rank_mapping_for_params_except_routed_expert(self):
        self.build_rank_mapping_two_stage(add_recv_actor_fn=None)

    def routed_experts_filter(self, name_list: List[str]):
        filted_names = [name for name in name_list if 'mlp.experts' in name]
        return filted_names

    def params_except_routed_expert_filter(self, name_list: List[str]):
        filted_names = [name for name in name_list if 'mlp.experts' not in name]
        return filted_names

    def clear_cache(self, sorted_send_actors_list=None, rank_mapping_list=None):
        if sorted_send_actors_list is None:
            sorted_send_actors_list = [
            self.sorted_send_actors,
            self.sorted_send_actors_stage2,
            self.send_actors_to_regroup_routed_experts,
            self.sorted_send_actors_for_routed_experts
        ]
        if rank_mapping_list is None:
            rank_mapping_list = [
                self.send_recv_actor_mappings,
                self.send_recv_actor_mappings_stage2,
                self.send_recv_actor_mappings_for_routed_experts
            ]

        self._clear_sync_send_recv_parameters(rank_mapping_list)
        self._clear_send_recv_param_names()
        self._clear_sorted_send_actors(sorted_send_actors_list)

    def warmup_groups(self):

        def warmup_tasks_func(task):
            actors = task.actors
            group = task.group
            refs = []
            refs.append(actors[0].broadcast_dummy_tensor_send.remote(0, group))
            for actor in actors[1:]:
                refs.append(actor.broadcast_dummy_tensor_recv.remote(0, group))
            future.wait(refs)

        tasks = []
        actors_set = set()
        for group_name, actors in self.groups2actors.items():
            # filter actors if the same collective ring
            actor_ids = [self.actor2rank[actor] for actor in actors]
            key = tuple(sorted(actor_ids))
            if key not in actors_set:
                tasks.append(CollectiveTask(actors, group_name))
        parallel_execute_collective_tasks(tasks, warmup_tasks_func)

    def _synchronize_all_moe_parameters(self, requires_grad=None, validate=False, dryrun=False):
        self.check_and_setup_collective_group()

        send_actors_list : List = [
            self.sorted_send_actors,
            self.sorted_send_actors_stage2
        ]
        actor_mappings_list : List = [
            self.send_recv_actor_mappings,
            self.send_recv_actor_mappings_stage2,
            self.send_actors_to_regroup_routed_experts,
        ]

        self.check_and_fuse_lora(self._enable_lora, actor_mappings_list)

        if self.concurrent_comm:
            assert self.dst_model.use_vllm_backend

            max_workers = self._calculate_max_workers(self.send_actors_to_regroup_routed_experts)
            if self._comm_type_to_regroup_routed_experts == ROUTED_EXPERT_REGROUPING_COMM_TYPE.ALLGATHER:
                # allgather routed experts only
                self.sync_allgather_multi_threads(
                    [self.send_actors_to_regroup_routed_experts],
                    max_workers=max_workers,
                    requires_grad=requires_grad,
                    group_name=self.group_name + "_allgather",
                    filter_fn=self.routed_experts_filter)
            elif self._comm_type_to_regroup_routed_experts == ROUTED_EXPERT_REGROUPING_COMM_TYPE.ALLTOALL:
                if not dryrun:
                    logger.info("start to alltoall router experts. ")
                    start_time = time.time()
                    # alltoall routed experts only
                    self.sync_alltoall_multi_threads(
                        [self.send_actors_to_regroup_routed_experts],
                        max_workers=max_workers,
                        requires_grad=requires_grad,
                        filter_fn=self.routed_experts_filter)
                    logger.info("complete to alltoall router experts using {time.time()-start_time:.2f} seconds ")
            # sync everything to inference model
            if self.tp_num_mapping == 1:
                logger.info("start to sync all moe experts")
                send_actors_list = [self.sorted_send_actors]
                actor_mappings_list = [self.send_recv_actor_mappings]
                self._multi_thread_sync_for_tp_num_mapping_eq_1(
                    send_actors_list,
                    actor_mappings_list,
                    requires_grad=requires_grad,
                    filter_fn=None,
                    param_group="default",
                    dryrun=dryrun
                )
                logger.info("complete to sync all moe experts")

            elif self.tp_num_mapping > 1:
                # First, synchronize routed experts.
                logger.info("start to sync routed expert weights.")
                start_time = time.time()
                self._synchronize_routed_experts(requires_grad=requires_grad, validate=validate, dryrun=dryrun)
                logger.info(f"complete to sync routed expert weights. [stage1-1] using {time.time()-start_time:.2f} seconds")
                self.clear_cache(
                    sorted_send_actors_list = [
                        self.send_actors_to_regroup_routed_experts,
                        self.sorted_send_actors_for_routed_experts
                    ],
                    rank_mapping_list=[
                        self.send_recv_actor_mappings_for_routed_experts
                    ]
                )

                # Then, synchronize parameters except routed experts
                logger.info("start to sync parameters except routed eperts.")
                self._synchronize_params_except_routed_experts(requires_grad=requires_grad, validate=validate, dryrun=dryrun)
                logger.info("complete to sync parameters except routed experts.")

                self.reset_synchronizer()

                self.clear_cache(
                    sorted_send_actors_list = [
                        self.sorted_send_actors,
                        self.sorted_send_actors_stage2,
                    ],
                    rank_mapping_list = [
                        self.send_recv_actor_mappings,
                        self.send_recv_actor_mappings_stage2
                    ]
                )
            else:
                raise NotImplementedError(
                    f"ChatLearn does not support synchronizing from larger tp size ({self.num_src_tensor_parallel})"
                    f"to smaller tp size ({self.num_dst_tensor_parallel}) currently."
                )

        else:
            raise NotImplementedError(
                "ChatLearn supports only concurrent_comm for training models with HEP enabled and inference with vLLM"
            )

        self.check_and_unfuse_lora(self._enable_lora, actor_mappings_list)

        self.validate_sync_results_parallel(actor_mappings_list, requires_grad, validate)

        self.check_and_destroy_collective_group()

        self.reset_synchronizer()

        logger.info(f"Group {self.group_name} sync all parameters done, comm_type {self._comm_type}")

    def _synchronize_routed_experts(self, requires_grad=None, validate=False, dryrun=False):
        self.check_and_setup_collective_group()

        self.check_and_fuse_lora(self._enable_lora, self.send_recv_actor_mappings_for_routed_experts)
        send_actors_list : List = []
        actor_mappings_list : List = []
        if self.concurrent_comm:
            send_actors_list = [self.sorted_send_actors_for_routed_experts]
            actor_mappings_list = [self.send_recv_actor_mappings_for_routed_experts]

            self._multi_thread_sync_for_tp_num_mapping_eq_1(
                send_actors_list,
                actor_mappings_list,
                requires_grad=requires_grad,
                filter_fn=self.routed_experts_filter,
                param_group="routed",
                dryrun=dryrun,
            )
        else:
            actor_mappings_list = [self.send_recv_actor_mappings_for_routed_experts]
            self._single_thread_sync(
                self.send_recv_actor_mappings_for_routed_experts,
                requires_grad=requires_grad,
                filter_fn=self.routed_experts_filter,
                param_group="routed",
            )

        assert len(actor_mappings_list) >= 1

        self.check_and_unfuse_lora(self._enable_lora, self.send_recv_actor_mappings_for_routed_experts)

        self.validate_sync_results_parallel(
            actor_mappings_list,
            requires_grad,
            validate,
            filter_fn=self.routed_experts_filter,
            param_group="routed"
        )

        self.check_and_destroy_collective_group()

        logger.info(f"Group {self.group_name} sync all parameters done, comm_type {self._comm_type}")

    def _synchronize_params_except_routed_experts(self, requires_grad=None, validate=False, dryrun=False):
        self.check_and_setup_collective_group()

        self.check_and_fuse_lora(self._enable_lora, self.send_recv_actor_mappings)

        send_actors_list : List = []
        actor_mappings_list : List = []
        if self.concurrent_comm:
            if self.tp_num_mapping > 1:
                send_actors_list = [self.sorted_send_actors, self.sorted_send_actors_stage2]
                actor_mappings_list = [self.send_recv_actor_mappings, self.send_recv_actor_mappings_stage2]
                self._multi_thread_sync_for_tp_num_mapping_gt_1(
                    send_actors_list,
                    actor_mappings_list,
                    requires_grad=requires_grad,
                    filter_fn=self.params_except_routed_expert_filter,
                    param_group="except_routed",
                    dryrun=dryrun
                )
            else:
                send_actors_list = [self.sorted_send_actors]
                actor_mappings_list = [self.send_recv_actor_mappings]
                self._multi_thread_sync_for_tp_num_mapping_eq_1(
                    send_actors_list,
                    actor_mappings_list,
                    requires_grad=requires_grad,
                    filter_fn=self.params_except_routed_expert_filter,
                    param_group="except_routed"
                )
        else:
            actor_mappings_list = [self.send_recv_actor_mappings]
            self._single_thread_sync(
                actor_mappings_list,
                requires_grad=requires_grad,
                filter_fn=self.params_except_routed_expert_filter,
                param_group="except_routed"
            )

        self.check_and_unfuse_lora(self._enable_lora, self.send_recv_actor_mappings)

        self.validate_sync_results_parallel(
            actor_mappings_list,
            requires_grad,
            validate,
            filter_fn=self.params_except_routed_expert_filter,
            param_group="except_routed"
        )

        self.check_and_destroy_collective_group()

        logger.info(f"Group {self.group_name} sync all parameters done, comm_type {self._comm_type}")

    def sync(self, requires_grad=None, validate=False, dryrun=False):
        if self.dst_model.use_vllm_backend:
            self.recover_synchronizer()
            self._synchronize_all_moe_parameters(requires_grad=requires_grad, validate=validate, dryrun=dryrun)
        else:
            if self.ep_num_mapping == 1 and self.tp_num_mapping == 1:
                # synchronization is the same as base class when applying Qwen + Qwen
                super().sync(requires_grad, validate)
                return

            self.recover_synchronizer()

            # First, synchronize routed experts.
            self._synchronize_routed_experts(requires_grad=requires_grad, validate=validate)

            self.clear_cache(
                sorted_send_actors_list = [
                    self.send_actors_to_regroup_routed_experts,
                    self.sorted_send_actors_for_routed_experts
                ],
                rank_mapping_list=[
                    self.send_recv_actor_mappings_for_routed_experts
                ]
            )

            # Then, synchronize parameters except routed experts
            self._synchronize_params_except_routed_experts(requires_grad=requires_grad, validate=validate)

            self.reset_synchronizer()

            self.clear_cache(
                sorted_send_actors_list = [
                    self.sorted_send_actors,
                    self.sorted_send_actors_stage2,
                ],
                rank_mapping_list = [
                    self.send_recv_actor_mappings,
                    self.send_recv_actor_mappings_stage2
                ]
            )
