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

import importlib
import concurrent.futures
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

from tqdm import tqdm

from chatlearn.launcher.initialize import patch_ray
from chatlearn.utils import future
from chatlearn.utils import utils
from chatlearn.utils.constant import LORA_WEIGHT_PREFIX
from chatlearn.utils.constant import PARAM_SYNC_COMM_TYPE
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.utils.utils import execute_in_parallel

vllm_exist = importlib.util.find_spec("vllm")
if vllm_exist:
    from chatlearn.models.vllm_module import VLLMModule


patch_ray()


class ParameterSyncGroup:
    """ParameterSyncGroup"""

    def __init__(self, src_model, dst_model, group_name, frequency, error_signal):
        self.src_model = src_model
        self.dst_model = dst_model
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
        self._num_src_tensor_parallel = None
        self._num_dst_tensor_parallel = None
        self._send_recv_param_names = {}
        self._actor2pipe = {}
        self._actor2tp = {}
        self._actor2dp = {}
        self._comm_type = get_args().runtime_args.param_sync_comm_type
        if src_model.colocate_with(dst_model) and self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
            if self.num_src_tensor_parallel % 2 == 1 and self.num_dst_tensor_parallel % 2 == 1:
                logger.warning("Only support PARAM_SYNC_COMM_TYPE.BROADCAST when TP SIZE is even number, use P2P instead")
                self._comm_type = PARAM_SYNC_COMM_TYPE.P2P
        self.setup_collective_group()
        self.num_mapping = self.num_dst_tensor_parallel // self.num_src_tensor_parallel
        if self.num_mapping == 1:
            self.build_rank_mapping()
        else:
            self.build_rank_mapping_two_stage()

        self.enable_coalesce_param = get_args().runtime_args.coalesce_param
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
        logger.debug(f"build rank mapping from {src_rank} to {dst_rank}, from gpu {src_gpu} to {dst_gpu}, " + \
                     f"from pipe_stage {src_pp_rank} to {dst_pp_rank}, " + \
                     f"from tp rank {src_tp_rank} to {dst_tp_rank}")
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

    def build_rank_mapping(self):
        # setup rank mapping for src parameter and dst parameter
        # get rank for one src_model, without model replicas
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

        assert len(src_ranks[0]) % len(dst_ranks[0]) == 0, \
            f"src training model ranks should be times of dst ranks, but got {len(src_ranks[0])} and {len(dst_ranks[0])}"
        if self.src_model.colocate_with(self.dst_model) and self.num_src_tensor_parallel % 2 == 1:
            replica_rank_iter = cycle(reversed(src_ranks))
        else:
            replica_rank_iter = cycle(iter(src_ranks))
        logger.debug(f"src_ranks: {src_ranks}")
        logger.debug(f"dst_ranks: {dst_ranks}")
        assert self.num_src_tensor_parallel == self.num_dst_tensor_parallel, \
            "currently we require the tensor_model_parallel_size to be the same between " + \
            f"src model {self.src_model.name}(TP={self.num_src_tensor_parallel}) and " + \
            f"dst model {self.dst_model.name}(TP={self.num_dst_tensor_parallel})"
        assert self.num_src_pipeline_stage % self.num_dst_pipeline_stage == 0

        def split_ranks_by_tp_size(ranks, tp_size):
            return [ranks[i:i + tp_size] for i in range(0, len(ranks), tp_size)]

        for dst_replica_ranks in dst_ranks:
            src_replica_ranks = next(replica_rank_iter)
            src_replica_ranks_group = split_ranks_by_tp_size(src_replica_ranks, self.num_src_tensor_parallel)
            dst_replica_ranks_group = split_ranks_by_tp_size(dst_replica_ranks, self.num_src_tensor_parallel)
            pipe_map_interval = self.num_src_pipeline_stage // self.num_dst_pipeline_stage
            for i, src_tp_group in enumerate(src_replica_ranks_group):
                j = i // pipe_map_interval
                for src_rank, dst_rank in zip(src_tp_group, dst_replica_ranks_group[j]):
                    self.add_recv_actor(src_rank, dst_rank)

    def build_rank_mapping_two_stage(self):
        # setup rank mapping for src parameter and dst parameter
        # get rank for one src_model, without model replicas
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

        def split_ranks_by_tp_size(ranks, tp_size):
            return [ranks[i:i + tp_size] for i in range(0, len(ranks), tp_size)]

        pair_list = []
        p2p_list = []
        for d_i, dst_replica_ranks in enumerate(dst_ranks):
            src_replica_ranks = next(replica_rank_iter)
            src_replica_ranks_group = split_ranks_by_tp_size(src_replica_ranks, self.num_src_tensor_parallel)
            dst_replica_ranks_group = split_ranks_by_tp_size(dst_replica_ranks, self.num_dst_tensor_parallel)
            logger.debug(f"src_replica_ranks_group: {src_replica_ranks_group}")
            logger.debug(f"dst_replica_ranks_group: {dst_replica_ranks_group}")
            pipe_map_interval = self.num_src_pipeline_stage // self.num_dst_pipeline_stage

            assert pipe_map_interval >= 1, \
                f"dst_pp expected to divide src_pp, while src_pp {self.num_src_pipeline_stage} and dst_pp {self.num_dst_pipeline_stage}"

            # stage 1: comm pairs that broadcast params from trainer to inference model
            # Each rank in trainer holds weights for num_mapping ranks in inference model.
            # For example: trainer_tp = 2, inference_tp = 4 => num_mapping = inference_tp // trainer_tp = 2
            # Weight mapping from training to inference:
            #   [0] -> [0', 1']
            #   [1] -> [2', 3']
            # To avoid p2p communication on the same gpu, we only broadcast params to first rank in weight_mapping_group.
            # Comm mapping from training to inference:
            #   [0] -> [0']
            #   [1] -> [2']
            for i, src_tp_group in enumerate(src_replica_ranks_group):
                j = i // pipe_map_interval
                if self.num_mapping == 1:
                    start =  0
                else:
                    mod_i = (i + d_i) % self.num_mapping
                    start = mod_i if i < self.num_mapping else (self.num_mapping - mod_i - 1) % self.num_mapping
                for s_idx, src_rank in enumerate(src_tp_group):
                    offset = s_idx * self.num_mapping + start

                    dst_rank = dst_replica_ranks_group[j][offset]
                    self.add_recv_actor(src_rank, dst_rank)
                    pair_list.append((src_rank, dst_rank))
                if pipe_map_interval == 1:
                    break

            # stage 2: comm pairs that broadcast params from first rank to the other ranks for each weight_mapping_group
            # Comm mapping in each weight_mapping_group of inference:
            #   [0'] -> [1']
            #   [2'] -> [3']
            recv_ranks = [pair[1] for pair in pair_list]
            def p2p_pair_grouping(tuples):
                for s_idx, src_rank in enumerate(tuples):
                    for d_idx, dst_rank in enumerate(tuples):
                        if s_idx == d_idx or src_rank not in recv_ranks:# pylint: disable=cell-var-from-loop
                            continue
                        self.add_recv_actor_stage2(src_rank, dst_rank)
                        p2p_list.append((src_rank, dst_rank))
            for dst_tp_group in dst_replica_ranks_group:
                dst_tp_group = split_ranks_by_tp_size(dst_tp_group, self.num_mapping)
                for tuples in dst_tp_group:
                    p2p_pair_grouping(tuples)

        logger.debug(f"comm pair_list <train_rank, inference_rank>: {pair_list}")
        logger.debug(f"comm p2p_list <inference_rank, inference_rank>: {p2p_list}")

    def _get_dst_name(self, src_name, src_prefix, dst_prefix):
        if src_prefix:
            dst_name = src_name[len(src_prefix):]
        else:
            dst_name = dst_prefix + src_name
        return dst_name

    def validate_sync_results(self, send_actor, recv_actor, requires_grad):

        def validate():
            # check the value of src model and tgt model
            src_names, dst_names = self.set_sync_param_names(send_actor, recv_actor, requires_grad)
            pipe_stage = self.get_actor_pipe_rank(send_actor)
            future.wait([send_actor.reset_sync_parameters.remote(src_names, pipe_stage),
                         recv_actor.reset_sync_parameters.remote(dst_names, pipe_stage)])
            src_names, dst_names = future.get([send_actor.get_parameter_to_sync_names.remote(pipe_stage),
                                               recv_actor.get_parameter_to_sync_names.remote(pipe_stage)])
            assert len(src_names) == len(dst_names)
            names = list(zip(src_names, dst_names))
            for src_name, dst_name in tqdm(names):
                src_tensor, dst_tensor = future.get([send_actor.get_parameter_to_sync.remote(src_name, pipe_stage, True),
                                                     recv_actor.get_parameter_to_sync.remote(dst_name, pipe_stage, True)])
                assert src_tensor.shape == dst_tensor.shape, \
                    f"after weight sync {src_name}: {src_tensor.shape} and {dst_name}: {dst_tensor.shape} do not match"
                assert (src_tensor == dst_tensor).all(), \
                    f"after weight sync {src_name}: {src_tensor} and {dst_name}: {dst_tensor} do not match"
            return True

        logger.info("Going to validate transmitted tensors...")
        validate()
        logger.info("Validation passed!")

    def set_sync_param_names_stage2(self, send_actor, recv_actor, to_rank, requires_grad):
        send_names, _ = self.set_sync_param_names(send_actor, send_actor, requires_grad)
        refs = []
        refs.append(send_actor.set_send_parameters.remote(send_names, self.get_actor_pipe_rank(send_actor)))
        refs.append(recv_actor.set_recv_parameters.remote(to_rank, send_names, self.get_actor_pipe_rank(recv_actor)))
        future.get(refs)
        return send_names, send_names

    def sync_broadcast_two_stage(self, actors, group_name, requires_grad=None, stage2=False):
        send_actor = actors[0]
        for rank, recv_actor in enumerate(actors[1:]):
            if stage2:
                self.set_sync_param_names_stage2(send_actor, recv_actor, self.actor2rank[recv_actor], requires_grad)
            else:
                self.set_sync_param_names(send_actor, recv_actor, requires_grad)
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
                    ele_buffer_num = 1 if send_param_num == recv_param_num else self.num_mapping
                    buffer_num[recv_name_and_shape[0]] = ele_buffer_num
                    tp_division[send_name_and_shape[0]] = ele_buffer_num
                refs = []
                refs.append(recv_actor.set_num_mapping.remote(self.num_mapping))
                refs.append(recv_actor.set_buffer_num.remote(buffer_num))
                refs.append(send_actor.set_num_mapping.remote(self.num_mapping))
                refs.append(send_actor.set_tp_division.remote(tp_division))
                future.get(refs)

        assert self.enable_coalesce_param
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

    def sync_broadcast(self, actors, group_name, requires_grad=None):
        send_actor = actors[0]
        for recv_actor in actors[1:]:
            self.set_sync_param_names(send_actor, recv_actor, requires_grad)
        pipe_stage = self.get_actor_pipe_rank(send_actor)
        assert self.enable_coalesce_param
        refs = []
        for rank, actor in enumerate(actors):
            ref = actor.broadcast_parameter.remote(rank, 0, group_name, pipe_stage)
            refs.append(ref)
        future.wait(refs, return_output=True)

    def _sync_send_recv(self, send_actor, recv_actor, requires_grad=None):
        src_names, dst_names = self.set_sync_param_names(send_actor, recv_actor, requires_grad)
        pipe_stage = self.get_actor_pipe_rank(send_actor)
        is_the_same_gpu = self.is_same_gpu(send_actor, recv_actor)

        if self.enable_coalesce_param:
            if is_the_same_gpu:
                name2ref = send_actor.ray_put_parameter.remote(None, self.group_name, pipe_stage)
                recv_ref = recv_actor.ray_get_parameter.remote(None, self.group_name, name2ref, pipe_stage)
                future.get(recv_ref)
            else:
                send_ref = send_actor.send_parameter.remote(None, self.actor2rank[recv_actor], self.group_name, pipe_stage)
                recv_ref = recv_actor.recv_parameter.remote(None, self.actor2rank[send_actor], self.group_name, pipe_stage)
                future.get([send_ref, recv_ref])
            logger.debug(f"sync all parameters from {send_actor} to {recv_actor}")
        else:
            dst_names_ref = future.get(recv_actor.get_parameter_names.remote(requires_grad=requires_grad))
            src_prefix, dst_prefix = self.set_model_prefix(src_names, dst_names_ref)
            for send_name, dst_name in zip(src_names, dst_names):
                dst_name = self._get_dst_name(send_name, src_prefix, dst_prefix)
                recv_tensor_exist = future.get(recv_actor.exist_parameter.remote(dst_name))
                if not recv_tensor_exist:
                    logger.info(f"recv tensor {dst_name} not exists")
                    all_dst_layer_names = future.get(recv_actor.get_parameter_names.remote())
                    raise Exception(
                        f"recv tensor {dst_name} not exists, while recv model has following layers {all_dst_layer_names}")
                if is_the_same_gpu:
                    raise Exception("In the single gpu case, enable_coalesce_param must be True, not False.")
                send_ref = send_actor.send_parameter.remote(send_name, self.actor2rank[recv_actor], self.group_name)
                recv_ref = recv_actor.recv_parameter.remote(dst_name, self.actor2rank[send_actor], self.group_name)
                future.get([send_ref, recv_ref])
            logger.debug(f"sync all parameters from {send_actor} to {recv_actor}, total param num {len(src_names)}")

    def sync_send_recv(self, send_actor, recv_actor, requires_grad=None):
        try:
            self._sync_send_recv(send_actor, recv_actor, requires_grad)
        except Exception:
            future.get(self.error_signal.set.remote(traceback.format_exc()))

    def set_model_prefix(self, src_names, dst_names):
        dst_prefix = None
        src_prefix = None
        for sname in src_names:
            for dname in dst_names:
                if sname in dname:
                    prefix = dname[:dname.index(sname)]
                    dst_prefix = prefix
                    return src_prefix, dst_prefix
                elif dname in sname:
                    prefix = sname[:sname.index(dname)]
                    src_prefix = prefix
                    return src_prefix, dst_prefix
        if dst_prefix is None and src_prefix is None:
            raise RuntimeError("Cannot find prefix")
        return src_prefix, dst_prefix

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

    def get_actor_dp_rank(self, actor):
        def inner_func():
            return future.get(actor.get_data_parallel_rank.remote())
        return utils.get_or_cache(self._actor2dp, actor, inner_func)

    def _set_sync_param_names(self, send_actor, recv_actor, requires_grad=None):
        if requires_grad is None:
            requires_grad = True
        if self._enable_lora:
            # TODO(jiangle.jl): support freeze layer.
            requires_grad = False
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

        if vllm_exist and isinstance(self.dst_model.replicas[0].model, VLLMModule):
            src_pipe_stage = self.get_actor_pipe_rank(send_actor)
            src_names, dst_names = future.get(recv_actor.map_src_to_dst.remote(src_names, self.num_src_pipeline_stage, src_pipe_stage))
            concat_params_dict = future.get(recv_actor.get_concat_params_dict.remote())
            future.get(send_actor.set_concat_params_dict.remote(concat_params_dict))
            to_fix_act_ordering_dict = future.get(recv_actor.get_to_fix_act_ordering_dict.remote())
            future.get(send_actor.set_to_fix_act_ordering_dict.remote(to_fix_act_ordering_dict))
            to_fix_qkv_ordering_dict = future.get(recv_actor.get_to_fix_qkv_ordering_dict.remote())
            future.get(send_actor.set_to_fix_qkv_ordering_dict.remote(to_fix_qkv_ordering_dict))
            to_fix_qkv_ordering_func = future.get(recv_actor.get_to_fix_qkv_ordering_func.remote())
            future.get(send_actor.set_to_fix_qkv_ordering_func.remote(to_fix_qkv_ordering_func))
        else:
            dst_names_ref = future.get(recv_actor.get_parameter_names.remote(requires_grad=False))
            src_prefix, dst_prefix = self.set_model_prefix(src_names, dst_names_ref)
            dst_names = [self._get_dst_name(name, src_prefix, dst_prefix) for name in dst_names]
        self.check_param_names(send_actor, recv_actor, src_names, dst_names)
        if self.num_mapping > 1:
            key = (recv_actor, recv_actor)
            if key not in self._send_recv_param_names:
                self._send_recv_param_names[key] = (dst_names, dst_names)
            else:
                dst_names0 = self._send_recv_param_names[key][0]
                dst_names0 += dst_names
                self._send_recv_param_names[key] = (dst_names0, dst_names0)
        if not (vllm_exist and isinstance(self.dst_model.replicas[0].model, VLLMModule)):
            pipe_stage = self.get_actor_pipe_rank(send_actor)
            refs = []
            refs.append(send_actor.set_sync_parameters.remote(src_names, pipe_stage))
            refs.append(recv_actor.set_sync_parameters.remote(dst_names, pipe_stage))
            future.get(refs)
        return src_names, dst_names

    def set_sync_param_names(self, send_actor, recv_actor, requires_grad=None):
        src_names, dst_names = utils.get_or_cache(self._send_recv_param_names, (send_actor, recv_actor), \
            lambda: self._set_sync_param_names(send_actor, recv_actor, requires_grad))
        pipe_stage = self.get_actor_pipe_rank(send_actor)
        if vllm_exist and isinstance(self.dst_model.replicas[0].model, VLLMModule):
            refs = []
            refs.append(send_actor.reset_sync_parameters.remote(src_names, pipe_stage))
            refs.append(recv_actor.reset_sync_parameters.remote(dst_names, pipe_stage))
            future.get(refs)
        return src_names, dst_names

    def create_broadcast_group(self, send_actor, recv_actors, group_name=None):
        actor_groups = [send_actor]
        actor_groups.extend(recv_actors)
        dp = self.get_actor_dp_rank(send_actor)
        pp = self.get_actor_pipe_rank(send_actor)
        tp = self.get_actor_tp_rank(send_actor)
        group_name = self.group_name if group_name is None else group_name
        group_name = f"{group_name}_dp{dp}_pp{pp}_tp{tp}"
        if group_name not in self.collective_groups:
            refs = []
            for rank, actor in enumerate(actor_groups):
                ref = actor.setup_collective_group.remote(rank, len(actor_groups), "nccl", group_name)
                refs.append(ref)
            future.wait(refs)
            self.collective_groups.append(group_name)
        return actor_groups, group_name

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

    def sync_broadcast_multi_threads(self, sorted_send_actors, send_recv_actor_mappings, max_workers, requires_grad, group_name=None, stage2=False):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for send_actor in sorted_send_actors:
                recv_actors = send_recv_actor_mappings[send_actor]
                if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                    if stage2:
                        for idx, recv_actor in enumerate(recv_actors):
                            group_name_ = f"{group_name}_{idx}"
                            actor_groups, group_name = self.create_broadcast_group(send_actor, [recv_actor], group_name=group_name_)
                            futures.append(executor.submit(self.sync_broadcast_two_stage, actor_groups, group_name, requires_grad, stage2))
                    else:
                        actor_groups, group_name = self.create_broadcast_group(send_actor, recv_actors, group_name=group_name)
                        futures.append(executor.submit(self.sync_broadcast_two_stage, actor_groups, group_name, requires_grad, stage2))
                else:
                    raise RuntimeError("support p2p only for scenes that trainer_tp not equal to inference_tp.")
            for _future in concurrent.futures.as_completed(futures):
                try:
                    _future.result()
                except Exception as e:
                    raise RuntimeError(f"Parameter sync thread generated an exception: {e}") # pylint: disable=raise-missing-from
            concurrent.futures.wait(futures)

    def sync(self, requires_grad=None, validate=False):
        if not self._is_collective_group_created:
            # Re-create collective group only when it is destroyed before.
            assert self._free_sync_collective_group
            self.setup_collective_group()

        for send_actor in self.send_recv_actor_mappings:
            if self._enable_lora:
                ref = send_actor.fuse_lora_layer.remote()
                state = future.get([ref])
                assert state, "Check fuse lora layer fail."

        if self.concurrent_comm:
            sorted_send_actors = self.sort_send_actors(self.send_recv_actor_mappings, self.sorted_send_actors)
            max_workers = get_args().runtime_args.param_sync_max_workers
            if max_workers is None:
                max_workers = max(self.src_model.total_gpu // 8, 1)
            if max_workers == -1:
                if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                    max_workers = len(send_actors)
                else:
                    max_workers = len(send_actors) * len(self.send_recv_actor_mappings[send_actors[0]])

            if self.num_mapping > 1:
                # stage 1
                self.sync_broadcast_multi_threads(sorted_send_actors, self.send_recv_actor_mappings, max_workers, requires_grad, stage2=False)
                # stage 2
                sorted_send_actors = self.sort_send_actors(self.send_recv_actor_mappings_stage2, self.sorted_send_actors_stage2)
                max_workers = get_args().runtime_args.param_sync_max_workers
                if max_workers is None:
                    max_workers = max(self.dst_model.total_gpu // 8, 1)
                if max_workers == -1:
                    if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                        max_workers = len(sorted_send_actors)
                    else:
                        max_workers = len(sorted_send_actors) * len(self.send_recv_actor_mappings_stage2[sorted_send_actors[0]])
                self.sync_broadcast_multi_threads(
                    sorted_send_actors, self.send_recv_actor_mappings_stage2, max_workers, requires_grad, group_name="intra_comm", stage2=True)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for send_actor in sorted_send_actors:
                        recv_actors = self.send_recv_actor_mappings[send_actor]
                        if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                            actor_groups, group_name = self.create_broadcast_group(send_actor, recv_actors)
                            futures.append(executor.submit(self.sync_broadcast, actor_groups, group_name, requires_grad))
                        else:
                            for recv_actor in recv_actors:
                                futures.append(executor.submit(self.sync_send_recv, send_actor, recv_actor, requires_grad))
                    for _future in concurrent.futures.as_completed(futures):
                        try:
                            _future.result()
                        except Exception as e:
                            raise RuntimeError(f"Parameter sync thread generated an exception: {e}") # pylint: disable=raise-missing-from
                    concurrent.futures.wait(futures)
        else:
            for send_actor, recv_actors in self.send_recv_actor_mappings.items():
                if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                    actor_groups, group_name = self.create_broadcast_group(send_actor, recv_actors)
                    self.sync_broadcast(actor_groups, group_name, requires_grad)
                else:
                    for recv_actor in recv_actors:
                        self.sync_send_recv(send_actor, recv_actor, requires_grad)

        for send_actor in self.send_recv_actor_mappings:
            if self._enable_lora:
                ref = send_actor.unfuse_lora_layer.remote()
                state = future.get([ref])
                assert state, "Check unfuse lora layer fail."

        if self._debug or validate:
            args = []
            for send_actor, recv_actors in self.send_recv_actor_mappings.items():
                for recv_actor in recv_actors:
                    args.append((send_actor, recv_actor, requires_grad))
            execute_in_parallel(self.validate_sync_results, args)

        if self._free_sync_collective_group:
            self.destroy_collective_group()
            self._is_collective_group_created = False
            self.collective_groups = []
        logger.info(f"Group {self.group_name} sync all parameters done, comm_type {self._comm_type}")
