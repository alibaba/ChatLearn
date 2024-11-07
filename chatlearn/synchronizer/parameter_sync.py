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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import List, Dict

import torch
from tqdm import tqdm

from chatlearn.launcher.initialize import patch_ray
from chatlearn.utils import future
from chatlearn.utils import utils
from chatlearn.utils.constant import LORA_WEIGHT_PREFIX
from chatlearn.utils.constant import PARAM_SYNC_COMM_TYPE
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.utils.utils import execute_in_parallel
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
        self.setup_collective_group()

        self.setup_rank_mapping()

        self.concurrent_comm = get_args().runtime_args.concurrent_comm
        self._enable_lora = self.src_model.module_args.lora.enable_lora
        # sync every n episodes, n = 0 for no param sync
        self._frequency = frequency

        self._free_sync_collective_group = get_args().runtime_args.free_sync_collective_group
        self._is_collective_group_created = True
        self.collective_groups = []
        self.src_dp_size = future.get(self.src_model.replicas[0].all_actors[0].get_data_parallel_size.remote())
        self.sorted_send_actors = None
        self.sorted_send_actors_stage2 = None

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

    def add_recv_actor(self, src_rank, dst_rank):
        src_actor = self.src_model.get_actor(src_rank)
        self.actor2rank[src_actor] = src_rank
        dst_actor = self.dst_model.get_actor(dst_rank)
        self.actor2rank[dst_actor] = dst_rank

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
        self.actor2rank[src_actor] = src_rank
        dst_actor = self.dst_model.get_actor(dst_rank)
        self.actor2rank[dst_actor] = dst_rank

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

    def build_rank_mapping(self, add_recv_actor_fn=None):
        # setup rank mapping for src parameter and dst parameter
        # get rank for one src_model, without model replicas

        if add_recv_actor_fn is None:
            add_recv_actor_fn = self.add_recv_actor

        dst_dp_ranks = self.dst_model.all_ranks
        local_src_ranks = future.get(self.src_model.replicas[0].get_local_param_ranks())
        if local_src_ranks[0] is None or dst_dp_ranks is None:
            if self._debug:
                logger.warning(
                    f"DEBUG MODE! src_dp_ranks {local_src_ranks} or dst_dp_ranks: {dst_dp_ranks} is None, "
                    "make sure they have values in real application.")
                return
            else:
                raise Exception(f"src_dp_ranks {local_src_ranks} or dst_dp_ranks {dst_dp_ranks} should not be None")
        dp_rank_to_ranks = defaultdict(list)
        for local_ranks, dp_rank in local_src_ranks:
            dp_rank_to_ranks[dp_rank].append(local_ranks[dp_rank])
        src_dp_ranks = [i[1] for i in sorted(dp_rank_to_ranks.items())]

        assert len(src_dp_ranks[0]) % len(dst_dp_ranks[0]) == 0, \
            f"src training model ranks should be times of dst ranks, but got {len(src_dp_ranks[0])} and {len(dst_dp_ranks[0])}"
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
            pipe_map_interval = self.num_src_pipeline_stage // self.num_dst_pipeline_stage
            for i, src_tp_group in enumerate(src_replica_ranks_group):
                j = i // pipe_map_interval
                for src_rank, dst_rank in zip(src_tp_group, dst_replica_ranks_group[j]):
                    add_recv_actor_fn(src_rank, dst_rank)

    # pylint: disable=unused-argument
    def build_rank_mapping_for_ep(self, add_recv_actor_fn=None):
        # Currently, we do nothing for ep
        pass

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

        dst_ranks = self.dst_model.all_ranks
        local_src_ranks = future.get(self.src_model.replicas[0].get_local_param_ranks())
        if local_src_ranks[0] is None or dst_ranks is None:
            if self._debug:
                logger.warning(
                    f"DEBUG MODE! src_ranks {local_src_ranks} or dst_ranks: {dst_ranks} is None, make sure they have values in real application.")
                return
            else:
                raise Exception(f"src_ranks {local_src_ranks} or dst_ranks {dst_ranks} should not be None")
        dp_rank_to_ranks = defaultdict(list)
        for local_ranks, dp_rank in local_src_ranks:
            dp_rank_to_ranks[dp_rank].append(local_ranks[dp_rank])
        src_ranks = [i[1] for i in sorted(dp_rank_to_ranks.items())]

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
        for dst_replica_ranks in dst_ranks:
            src_replica_ranks = next(replica_rank_iter)
            src_replica_ranks_group = split_ranks_by_tp_and_ep_size(src_replica_ranks, self.num_src_tensor_parallel, self.num_src_expert_parallel)
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
                    offset = s_idx * self.tp_num_mapping + start
                    dst_rank = dst_replica_ranks_group[j][offset]
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

        logger.debug(f"comm pair_list <train_rank, inference_rank>: {pair_list}")
        logger.debug(f"comm p2p_list <inference_rank, inference_rank>: {p2p_list}")

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
            sorted_send_actors_list = [self.sorted_send_actors, self.sorted_send_actors_stage2]
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

    def _set_sync_param_names(self, send_actor, recv_actor, requires_grad=None, filter_fn=None, param_group="default"):
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

        src_names, dst_names = self.synchronizer.map_name_from_src_to_dst(send_actor, recv_actor, src_names, dst_names)
        future.wait(send_actor.set_synchronizer.remote(self.synchronizer))

        self.check_param_names(send_actor, recv_actor, src_names, dst_names)
        if param_group != "routed" and self.tp_num_mapping > 1:
            key = (recv_actor, recv_actor)
            if key not in self._send_recv_param_names:
                self._send_recv_param_names[key] = (dst_names, dst_names)
            else:
                dst_names0 = self._send_recv_param_names[key][0]
                dst_names0 += dst_names
                self._send_recv_param_names[key] = (dst_names0, dst_names0)
        elif param_group == "routed":
            # Do nothing becuase the routed experts are one-to-one mapped across training and inference currently.
            assert self.hep_num_mapping == 1, (
                "Currently, ChatLearn supports balanced hyper expert parallel size across training and inference only, "
                "i.e. training EP size * training TP size must be equal to inference EP size * inference TP size."
            )
        if not self.synchronizer.is_parameter_changed:
            pipe_stage = self.get_actor_pipe_rank(send_actor)
            refs = []
            refs.append(send_actor.set_sync_parameters.remote(src_names, pipe_stage))
            refs.append(recv_actor.set_sync_parameters.remote(dst_names, pipe_stage))
            future.get(refs)
        return src_names, dst_names

    def set_sync_param_names(self, send_actor, recv_actor, requires_grad=None, filter_fn=None, param_group="default"):
        src_names, dst_names = utils.get_or_cache(self._send_recv_param_names, (send_actor, recv_actor), \
            lambda: self._set_sync_param_names(send_actor, recv_actor, requires_grad, filter_fn, param_group))
        logger.debug(f"{self.actor2rank[send_actor]} -> {self.actor2rank[recv_actor]}: {src_names} -> {dst_names}")
        pipe_stage = self.get_actor_pipe_rank(send_actor)
        if self.synchronizer.is_parameter_changed:
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

    def sync_broadcast_multi_threads(
        self, sorted_send_actors, send_recv_actor_mappings, max_workers, requires_grad,
        group_name=None, stage2=False, filter_fn=None, param_group="default"):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for send_actor in sorted_send_actors:
                recv_actors = send_recv_actor_mappings[send_actor]
                if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                    if stage2:
                        for idx, recv_actor in enumerate(recv_actors):
                            group_name_with_idx = f"{group_name}_{idx}"
                            actor_groups, finalized_group_name = self.create_broadcast_group(
                                send_actor, [recv_actor], group_name=group_name_with_idx, param_group=param_group
                            )
                            futures.append(executor.submit(
                                self.sync_broadcast_two_stage, actor_groups, finalized_group_name, requires_grad, stage2, filter_fn, param_group
                            ))
                    else:
                        actor_groups, finalized_group_name = self.create_broadcast_group(
                            send_actor, recv_actors, group_name=group_name, param_group=param_group
                        )
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

    def check_and_fuse_lora(self, enable_lora, actor_mapping):
        assert isinstance(actor_mapping, Dict)
        for send_actor in actor_mapping:
            if enable_lora:
                ref = send_actor.fuse_lora_layer.remote()
                state = future.get([ref])
                assert state, "Check fuse lora layer fail."

    def check_and_unfuse_lora(self, enable_lora, actor_mapping):
        assert isinstance(actor_mapping, Dict)
        for send_actor in actor_mapping:
            if self._enable_lora:
                ref = send_actor.unfuse_lora_layer.remote()
                state = future.get([ref])
                assert state, "Check unfuse lora layer fail."

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
            execute_in_parallel(self.validate_sync_results, args)

    def _calculate_max_workers(self, sorted_send_actors, actor_mapping):
        max_workers = get_args().runtime_args.param_sync_max_workers
        if max_workers is None:
            max_workers = max(self.src_model.total_gpu // 8, 1)
        if max_workers == -1:
            if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                max_workers = len(sorted_send_actors)
            else:
                max_workers = len(sorted_send_actors) * len(actor_mappings[sorted_send_actors[0]])
        return max_workers

    def _multi_thread_sync_for_tp_num_mapping_gt_1(
        self,
        send_actors:List,
        actor_mappings:List,
        requires_grad=None,
        filter_fn=None,
        param_group="default"
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
        sorted_send_actors_stage1 = list(actor_mappings_stage1.keys())
        max_workers = self._calculate_max_workers(sorted_send_actors_stage1, actor_mappings_stage1)
        group_name = self.group_name + "_inter_comm"
        self.sync_broadcast_multi_threads(
            sorted_send_actors_stage1, actor_mappings_stage1, max_workers, requires_grad,
            group_name=group_name, stage2=False, filter_fn=filter_fn, param_group=param_group
        )
        # stage 2
        sorted_send_actors_stage2 = list(actor_mappings_stage2.keys())
        max_workers = self._calculate_max_workers(sorted_send_actors_stage2, actor_mappings_stage2)
        group_name = self.group_name + "_intra_comm"
        self.sync_broadcast_multi_threads(
            sorted_send_actors_stage2, actor_mappings_stage2, max_workers, requires_grad,
            group_name=group_name, stage2=True, filter_fn=filter_fn, param_group=param_group)

    def _multi_thread_sync_for_tp_num_mapping_eq_1(
        self, send_actors_list:List, actor_mappings_list:List,
        requires_grad=None, filter_fn=None, param_group="default"
    ):
        assert len(send_actors_list) == 1 and len(actor_mappings_list) == 1
        send_actors = send_actors_list[0]
        actor_mappings = actor_mappings_list[0]

        sorted_send_actors = self.sort_send_actors(actor_mappings, send_actors)
        max_workers = self._calculate_max_workers(sorted_send_actors, actor_mappings)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for send_actor in sorted_send_actors:
                recv_actors = actor_mappings[send_actor]
                if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                    actor_groups, finalized_group_name = self.create_broadcast_group(send_actor, recv_actors, param_group=param_group)
                    futures.append(executor.submit(
                        self.sync_broadcast, actor_groups, finalized_group_name, requires_grad, filter_fn=filter_fn, param_group=param_group
                    ))
                else:
                    for recv_actor in recv_actors:
                        futures.append(executor.submit(
                            self.sync_send_recv, send_actor, recv_actor, requires_grad, filter_fn=filter_fn, param_group=param_group
                        ))
            for _future in concurrent.futures.as_completed(futures):
                try:
                    _future.result()
                except Exception as e:
                    raise RuntimeError(f"Parameter sync thread generated an exception: {e}") # pylint: disable=raise-missing-from
            concurrent.futures.wait(futures)

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

    def sync(self, requires_grad=None, validate=False):
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

        logger.info(f"Group {self.group_name} sync all parameters done, comm_type {self._comm_type}")


class ParameterSyncGroupwithHEP(ParameterSyncGroup):
    """ParameterSyncGroup for Hyper Expert Parallel (HEP).
    
       Note that in HEP, EP size for routed experts is not equivalent to that for shared experts.
       For routed experts, the new EP size (we call it HEP size for clarification) = mpu.ep_size x mpu.tp_size.
       For shared experts, the EP size remains 1 because they cannot be parallelized in expert dimension.
       In this case, shared experts in HEP shares the same parallel dimension with other non-expert weights.
       Therefore, we manage only two seperate parameter sync groups for routed expert weigts and other weights.
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
        if self.ep_num_mapping == 1 and self.tp_num_mapping == 1:
            # In this special case, all parameters are mapped one by one
            self.build_rank_mapping()
        elif self.hep_num_mapping == 1:
            # In this case, routed experts are mapped one by one, while params except routed experts are split by TP.
            self.build_rank_mapping_for_routed_experts()
            self.build_rank_mapping_for_params_except_routed_expert()
        else:
            # We do not support other cases for HEP. Please note that tp_num_mapping > 1 with ep_num_mapping = 1 is also unsupported.
            raise NotImplementedError(
                "ChatLearn does not support inequivalent EP x TP between training and inference with Hyper Expert Parallel (HEP) enabled now. "
                f"Your current setting is EP{self.num_src_expert_parallel} TP{self.num_src_tensor_parallel} for training model {self.src_model.name} "
                f"and EP{self.num_dst_expert_parallel} TP{self.num_dst_tensor_parallel} for inference model {self.dst_model.name}."
            )

    def add_recv_actor_for_routed_experts(self, src_rank, dst_rank):
        src_actor = self.src_model.get_actor(src_rank)
        self.actor2rank[src_actor] = src_rank
        dst_actor = self.dst_model.get_actor(dst_rank)
        self.actor2rank[dst_actor] = dst_rank

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

    def _synchronize_routed_experts(self, requires_grad=None, validate=False):
        assert self.hep_num_mapping == 1, (
            "Currently, _synchronize_routed_experts requires EP x TP for src model is equal to that for dst model"
        )

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

    def _synchronize_params_except_routed_experts(self, requires_grad=None, validate=False):
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
                    param_group="except_routed"
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

    def sync(self, requires_grad=None, validate=False):
        if self.ep_num_mapping == 1 and self.tp_num_mapping == 1:
            super().sync(requires_grad, validate)
            return

        # First, synchronize routed experts.
        self._synchronize_routed_experts(requires_grad=requires_grad, validate=validate)

        self.clear_cache(
            sorted_send_actors_list = [
                self.sorted_send_actors_for_routed_experts
            ],
            rank_mapping_list=[
                self.send_recv_actor_mappings_for_routed_experts
            ]
        )

        # Then, synchronize parameters except routed experts
        self._synchronize_params_except_routed_experts(requires_grad=requires_grad, validate=validate)

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
