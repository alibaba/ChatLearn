# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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

import random
import threading
import traceback
from collections import defaultdict
from itertools import cycle

from tqdm import tqdm

from chatlearn import get_args
from chatlearn.launcher.initialize import patch_ray
from chatlearn.utils import future
from chatlearn.utils import utils
from chatlearn.utils.constant import LORA_WEIGHT_PREFIX
from chatlearn.utils.logger import logger

patch_ray()


class ParameterSyncGroup:
    """ParameterSyncGroup"""

    def __init__(self, src_model, dst_model, group_name, error_signal):
        self.src_model = src_model
        self.dst_model = dst_model
        self.group_name = group_name
        self.error_signal = error_signal
        self.send_recv_actor_mappings = defaultdict(list)
        self.recv_send_actor_mappings = defaultdict(list)
        self.actor2rank = {}
        self._debug = get_args().rlhf_args.debug
        self._num_src_pipeline_stage = None
        self._num_dst_pipeline_stage = None
        self._num_src_tensor_parallel = None
        self._num_dst_tensor_parallel = None
        self._dst_prefix = None
        self._src_prefix = None
        self._send_recv_param_names = {}
        self._actor2pipe = {}
        self._actor2tp = {}
        self._validate_params = {}
        self.setup_collective_group()
        self.build_rank_mapping()
        self.enable_coalesce_param = get_args().rlhf_args.coalesce_param
        self.concurrent_comm = get_args().rlhf_args.concurrent_comm
        self._enable_lora = self.src_model.module_args.lora.enable_lora

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
            logger.info(
                f"start setup_collective_group for {model.name}, group_name: {self.group_name}, world_size: {world_size}, rank_offset: {rank_offset}")
            for replica in model.replicas:
                refs += replica._setup_collective_group(rank_offset, world_size, self.group_name)
                rank_offset += replica.actor_num

        future.get(refs)
        logger.info(f"init collective group done for {self.group_name}")

    def destroy_collective_group(self):
        try:
            self.src_model.destroy_collective_group()
            self.dst_model.destroy_collective_group()
            logger.info(f"destroy_collective_group success for {self.group_name}")
        except Exception as e:
            logger.exception(f"destroy_collective_group fail for {self.group_name} {e}")

    def add_recv_actor(self, src_rank, dst_rank):
        src_actor = self.src_model.get_actor(src_rank)
        self.actor2rank[src_actor] = src_rank
        dst_actor = self.dst_model.get_actor(dst_rank)
        self.actor2rank[dst_actor] = dst_rank

        src_gpu = future.get(src_actor.get_visible_gpus.remote())
        dst_gpu = future.get(dst_actor.get_visible_gpus.remote())
        src_tp_rank = self.get_actor_tp_rank(src_actor)
        dst_tp_rank = self.get_actor_tp_rank(dst_actor)
        src_pp_rank = self.get_actor_pipe_rank(src_actor)
        dst_pp_rank = self.get_actor_pipe_rank(dst_actor)
        logger.info(f"build rank mapping from {src_rank} to {dst_rank}, from gpu {src_gpu} to {dst_gpu}, " + \
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

    def validate_sync_results(self, send_actor, recv_actor, src_names, dst_names):

        def validate():
            # check the value of src model and tgt model
            random_names = random.sample(list(zip(src_names, dst_names)), 5)
            for src_name, dst_name in tqdm(random_names):
                src_tensor = future.get(send_actor.get_parameter.remote(src_name))
                dst_tensor = future.get(recv_actor.get_parameter.remote(dst_name))
                assert (
                        src_tensor == dst_tensor).all(), f"after weight sync {src_name}: {src_tensor} and {dst_name}: {dst_tensor} do not match"
            return True

        if self._debug:
            logger.info("Going to validate transmitted tensors...")
            utils.get_or_cache(self._validate_params, (send_actor, recv_actor), validate)
            logger.info("Validation passed!")

    def _sync_send_recv(self, send_actor, recv_actor, requires_grad=None):
        src_names, dst_names = self.get_sync_param_names(send_actor, recv_actor, requires_grad)
        pipe_stage = self.get_actor_pipe_rank(send_actor)
        send_gpu = future.get(send_actor.get_visible_gpus.remote())
        recv_gpu = future.get(recv_actor.get_visible_gpus.remote())
        is_the_same_gpu = (len(send_gpu) == len(recv_gpu) and len(send_gpu) == 1 and send_gpu[0] == recv_gpu[0])

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
        self.validate_sync_results(send_actor, recv_actor, src_names, dst_names)

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
            dst_names = dst_src_mappings.keys()
            src_names = dst_src_mappings.values()
        else:
            src_names = dst_names = future.get(send_actor.get_parameter_names.remote(requires_grad=requires_grad))

        if self._enable_lora:
            src_names = [ele for ele in src_names if LORA_WEIGHT_PREFIX not in ele]
            dst_names = [ele for ele in dst_names if LORA_WEIGHT_PREFIX not in ele]

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

    def get_sync_param_names(self, send_actor, recv_actor, requires_grad=None):
        return utils.get_or_cache(self._send_recv_param_names, (send_actor, recv_actor), \
                                  lambda: self._set_sync_param_names(send_actor, recv_actor, requires_grad))

    def sync(self, requires_grad=None):
        threads = []
        use_threads = self.concurrent_comm
        for send_actor in self.send_recv_actor_mappings:
            if self._enable_lora:
                ref = send_actor.fuse_lora_layer.remote()
                state = future.get([ref])
                assert state, "Check fuse lora layer fail."
            recv_actors = self.send_recv_actor_mappings[send_actor]
            for recv_actor in recv_actors:
                if use_threads:
                    thread = threading.Thread(target=self.sync_send_recv, args=(send_actor, recv_actor, requires_grad))
                    threads.append(thread)
                else:
                    self.sync_send_recv(send_actor, recv_actor, requires_grad)
            if self._enable_lora:
                ref = send_actor.unfuse_lora_layer.remote()
                state = future.get([ref])
                assert state, "Check unfuse lora layer fail."
        if len(threads) > 0:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        logger.info("sync all parameters done")
