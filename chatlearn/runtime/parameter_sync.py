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
        self.actor2rank = {}
        self._debug = get_args().runtime_args.debug
        self._num_src_pipeline_stage = None
        self._num_dst_pipeline_stage = None
        self._num_src_tensor_parallel = None
        self._num_dst_tensor_parallel = None
        self._dst_prefix = None
        self._src_prefix = None
        self._send_recv_param_names = {}
        self._actor2pipe = {}
        self._actor2tp = {}
        self._actor2dp = {}
        self._validate_params = {}
        self._comm_type = get_args().runtime_args.param_sync_comm_type
        if src_model.colocate_with(dst_model) and self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
            if self.num_src_tensor_parallel % 2 == 1:
                logger.warning("Only support PARAM_SYNC_COMM_TYPE.BROADCAST when TP SIZE is even number, use P2P instead")
                self._comm_type = PARAM_SYNC_COMM_TYPE.P2P
        self.setup_collective_group()
        self.build_rank_mapping()
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

    def get_group_name(self, actors):
        return f"{self.group_name}_" + "_".join(str(self.actor2rank[actor]) for actor in actors)

    @property
    def frequency(self):
        return self._frequency

    def get_or_cache(self, actor, func_name):
        def inner_func():
            return future.get(getattr(getattr(actor, func_name), 'remote')())
        cached_name = "_actor2" + func_name
        if hasattr(self, cached_name):
            cached = getattr(self, cached_name)
        else:
            cached = {}
            setattr(self, cached_name, cached)
        return utils.get_or_cache(cached, actor, inner_func)

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
        assert src_tp_rank == dst_tp_rank, f"src_tp_rank {src_tp_rank} should be same as dst_tp_rank {dst_tp_rank}"
        self.send_recv_actor_mappings[src_actor].append(dst_actor)
        self.recv_send_actor_mappings[dst_actor].append(src_actor)

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

    def _get_dst_name(self, src_name):
        if self._src_prefix:
            dst_name = src_name[len(self._src_prefix):]
        else:
            dst_name = self._dst_prefix + src_name
        return dst_name

    def validate_sync_results(self, send_actor, recv_actor, requires_grad):
        src_names, dst_names = self.set_sync_param_names(send_actor, recv_actor, requires_grad)

        def validate():
            # check the value of src model and tgt model
            names = list(zip(src_names, dst_names))
            for src_name, dst_name in tqdm(names):
                src_tensor = future.get(send_actor.get_parameter.remote(src_name))
                dst_tensor = future.get(recv_actor.get_parameter.remote(dst_name))
                assert (
                        src_tensor == dst_tensor).all(), f"after weight sync {src_name}: {src_tensor} and {dst_name}: {dst_tensor} do not match"
            return True

        if self._debug:
            logger.info("Going to validate transmitted tensors...")
            utils.get_or_cache(self._validate_params, (send_actor, recv_actor), validate)
            logger.info("Validation passed!")

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
            for send_name, dst_name in zip(src_names, dst_names):
                dst_name = self._get_dst_name(send_name)
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
        for sname in src_names:
            for dname in dst_names:
                if sname in dname:
                    prefix = dname[:dname.index(sname)]
                    self._dst_prefix = prefix
                    return
                elif dname in sname:
                    prefix = sname[:sname.index(dname)]
                    self._src_prefix = prefix
                    return
        if self._dst_prefix is None and self._src_prefix is None:
            raise RuntimeError("Cannot find prefix")

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
            dst_src_mappings = future.get(send_actor.build_pipeline_layer_name_mapping.remote(
                                          self.num_dst_pipeline_stage, dst_pipe_rank, requires_grad=requires_grad))
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
            if self._dst_prefix is None and self._src_prefix is None:
                dst_names_ref = future.get(recv_actor.get_parameter_names.remote(requires_grad=False))
                self.set_model_prefix(src_names, dst_names_ref)

            dst_names = [self._get_dst_name(name) for name in dst_names]
        self.check_param_names(send_actor, recv_actor, src_names, dst_names)
        pipe_stage = self.get_actor_pipe_rank(send_actor)
        refs = []
        refs.append(send_actor.set_sync_parameters.remote(src_names, pipe_stage))
        refs.append(recv_actor.set_sync_parameters.remote(dst_names, pipe_stage))
        future.get(refs)
        return src_names, dst_names

    def set_sync_param_names(self, send_actor, recv_actor, requires_grad=None):
        return utils.get_or_cache(self._send_recv_param_names, (send_actor, recv_actor), \
                                  lambda: self._set_sync_param_names(send_actor, recv_actor, requires_grad))

    def create_broadcast_group(self, send_actor, recv_actors):
        actor_groups = [send_actor]
        actor_groups.extend(recv_actors)
        dp = self.get_actor_dp_rank(send_actor)
        pp = self.get_actor_pipe_rank(send_actor)
        tp = self.get_actor_tp_rank(send_actor)
        group_name = f"{self.group_name}_dp{dp}_pp{pp}_tp{tp}"
        if group_name not in self.collective_groups:
            refs = []
            for rank, actor in enumerate(actor_groups):
                ref = actor.setup_collective_group.remote(rank, len(actor_groups), "nccl", group_name)
                refs.append(ref)
            future.wait(refs)
            self.collective_groups.append(group_name)
        return actor_groups, group_name

    def sort_send_actors(self):
        if self.sorted_send_actors is not None:
            return self.sorted_send_actors
        dp2send_actors = defaultdict(list)
        for send_actor in self.send_recv_actor_mappings:
            dp2send_actors[self.get_actor_dp_rank(send_actor)].append(send_actor)
        for dp_rank in dp2send_actors:
            send_actors = dp2send_actors[dp_rank]
            dp2send_actors[dp_rank] = sorted(send_actors, key=lambda x: self.actor2rank[x])
        sorted_send_actors = []
        dp_rank = 0
        while len(sorted_send_actors) < len(self.send_recv_actor_mappings):
            sorted_send_actors.append(dp2send_actors[dp_rank].pop(0))
            dp_rank += 1
            # dp_rank not in dp2send_actors happens when inference replica number less than training replica number
            if dp_rank == self.src_dp_size or dp_rank not in dp2send_actors:
                dp_rank = 0
        assert len(self.send_recv_actor_mappings) == len(sorted_send_actors)
        self.sorted_send_actors = sorted_send_actors
        return sorted_send_actors

    def sync(self, requires_grad=None):
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
            sorted_send_actors = self.sort_send_actors()
            max_workers = get_args().runtime_args.param_sync_max_workers
            if max_workers is None:
                max_workers = max(self.src_model.total_gpu // 8, 1)
            if max_workers == -1:
                if self._comm_type == PARAM_SYNC_COMM_TYPE.BROADCAST:
                    max_workers = len(send_actors)
                else:
                    max_workers = len(send_actors) * len(self.send_recv_actor_mappings[send_actors[0]])
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

        if self._debug:
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
