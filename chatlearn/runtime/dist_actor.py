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
"""Dist Actor"""

from collections import defaultdict
import importlib
import inspect
from functools import partial

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from chatlearn.models.base_module import BaseModule
from chatlearn.utils import future
from chatlearn.utils.utils import parse_function_args

vllm_exist = importlib.util.find_spec("vllm")
if vllm_exist:
    from chatlearn.models.vllm_module import VLLMModule

RAY_REMOTE = "remote"


class DistActor:
    """Manage a collection of actors"""

    def __init__(self, model: BaseModule,
                 gpu_per_node,
                 error_signal,
                 port_manager,
                 replica_id=0,
                 storage=None):
        self.total_gpu = model.total_gpu
        self.total_cpu = model.total_cpu
        self.gpu_per_process = model.gpu_per_process
        self.num_gpu_per_replica = model.num_gpu_per_replica
        self.trainable = model.trainable
        self.gpu_per_node = gpu_per_node
        self.model = model
        self.all_actors = []
        self.replica_id = replica_id
        self._port_manager = port_manager
        self.name = self.model.name
        self.error_signal = error_signal
        self.storage = storage
        # ranks for model update
        self.all_ranks = None
        self._init_done = False
        self._placement_group = None
        self.rank_to_actors = {}

    @property
    def module_args(self):
        return self.model.module_args

    @property
    def runtime_args(self):
        return self.model.runtime_args

    @property
    def master(self):
        return self.all_actors[0]

    @property
    def tailer(self):
        return self.all_actors[-1]

    @property
    def actor_num(self):
        return len(self.all_actors)

    def _get_func_args(self, func_name):
        func = getattr(self.model, func_name)
        return parse_function_args(func)

    def preprocess_actors(self):
        self.add_remote_func()

    def add_remote_func(self):
        for func_name, _ in inspect.getmembers(self.master):
            # ray.actor.ActorMethod
            if func_name.startswith('_'):
                continue
            dist_call = partial(self.call_remote_funcs, func_name)
            setattr(self, func_name, dist_call)

    def call_actor_remote_func(self, actor, func_name, *args, **kwargs):
        func = getattr(actor, func_name)
        remote_func = getattr(func, RAY_REMOTE)
        res = remote_func(*args, **kwargs)
        return res

    def call_remote_funcs(self, func_name, *args, **kwargs):
        """
        Call remote functions for a collection of actors.
        """
        results = []
        for actor in self.all_actors:
            res = self.call_actor_remote_func(actor, func_name, *args, **kwargs)
            results.append(res)
        return results

    def create_actor(self, num_gpus, placement_group, group_index):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=group_index,
        )
        # use max_concurrency=1 to make sure only one task execute at one time
        actor = ray.remote(num_gpus=num_gpus, num_cpus=0)(self.model.__class__) \
            .options(scheduling_strategy=scheduling_strategy,
                     max_concurrency=1) \
            .remote(self.model.name, self.model.global_args, self.replica_id)
        actor.set_error_signal.remote(self.error_signal)
        actor.set_storage.remote(self.storage)
        self.all_actors.append(actor)

    def _setup_collective_group(self, rank_offset, world_size, group_name, backend="nccl"):
        refs = []
        all_ranks = []
        for i, actor in enumerate(self.all_actors):
            rank = i + rank_offset
            ref = actor.setup_collective_group.remote(
                rank=rank,
                world_size=world_size,
                backend=backend,
                group_name=group_name)
            refs.append(ref)
            all_ranks.append(rank)
            self.rank_to_actors[rank] = actor
        self.all_ranks = all_ranks
        return refs

    def _setup_ranks(self, rank_offset):
        all_ranks = []
        for i, actor in enumerate(self.all_actors):
            rank = i + rank_offset
            all_ranks.append(rank)
            self.rank_to_actors[rank] = actor
        self.all_ranks = all_ranks

    def terminate(self):
        # terminate when catching exceptions
        for actor in self.all_actors:
            ray.kill(actor)

    @property
    def placement_group(self):
        return self._placement_group

    @placement_group.setter
    def placement_group(self, pg):
        self._placement_group = pg

    def group_dist_actors_by_tp_rank(self):
        self.dp_rank_to_actors = defaultdict(list)
        self.data_parallel_size = future.get(self.all_actors[0].get_data_parallel_size.remote())
        if self.data_parallel_size is None:
            self.data_parallel_size = 1
        dp_ranks = future.wait([actor.get_data_parallel_rank.remote() for actor in self.all_actors], return_output=True)
        for actor, dp_rank in zip(self.all_actors, dp_ranks):
            self.dp_rank_to_actors[dp_rank].append(actor)

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self):
        return f'<{self.__class__.__name__}({self.name}) object at {hex(id(self))}>'


class DistTorchActor(DistActor):
    """DistTorchActor"""

    def reorder_actors(self, actors, revert_placement=False):
        gpu_per_node = min(self.gpu_per_node, self.model.num_gpu_per_replica)
        ordered_actors = []
        count = 0
        actor_gpus = []
        for actor in actors:
            gpus = future.get(actor.get_visible_gpus.remote())
            count += len(gpus)
            actor_gpus.append((actor, gpus))
            if count == gpu_per_node:
                actor_gpus.sort(key=lambda x: x[1][0])
                if revert_placement:
                    actor_gpus.reverse()
                ordered_actors += [a[0] for a in actor_gpus]
                actor_gpus = []
                count = 0
        return ordered_actors

    def set_dist_env(self, revert_placement=False):
        self.all_actors = self.reorder_actors(self.all_actors, revert_placement)
        master_addr = future.get(self.master.get_address.remote())
        master_port = future.get(self._port_manager.get_free_port.remote(master_addr))

        world_size = self.actor_num
        env_config = {"MASTER_ADDR": master_addr, "MASTER_PORT": master_port, "WORLD_SIZE": world_size}
        ret = []
        for rank, actor in enumerate(self.all_actors):
            env_config["RANK"] = rank
            if self.model.gpu_per_process == 1:
                local_rank = 0
            else:
                local_rank = rank % self.model.gpu_per_process
            env_config["LOCAL_RANK"] = local_rank
            ret.append(actor.set_env.remote(env_config))
        status = sum(future.get(ret))
        assert status == world_size


class DistModel:
    """DistModel"""

    def __init__(self):
        self.replicas = []
        self.name = None
        self.rank_to_actors = {}
        self.register_serial_func()
        self.register_func()
        self._is_colocate = False
        self._colocate_models = None

    def add_replica(self, replica):
        self.replicas.append(replica)
        self.name = replica.name

    @property
    def trainable(self):
        return self.replicas[0].trainable

    @property
    def module_args(self):
        return self.replicas[0].module_args

    @property
    def actor_num(self):
        return sum(len(dist_actor.all_actors) for dist_actor in self.replicas)

    @property
    def num_replica(self):
        return len(self.replicas)

    @property
    def total_gpu(self):
        return self.replicas[0].total_gpu

    @property
    def total_cpu(self):
        return self.replicas[0].total_cpu

    @property
    def num_gpu_per_replica(self):
        return self.replicas[0].num_gpu_per_replica

    @property
    def gpu_per_process(self):
        return self.replicas[0].gpu_per_process

    @property
    def is_colocate(self):
        return self._is_colocate

    @is_colocate.setter
    def is_colocate(self, flag):
        self._is_colocate = flag

    def get_actor(self, rank):
        # given rank, return the actor
        for dist_actor in self.replicas:
            if rank in dist_actor.rank_to_actors:
                return dist_actor.rank_to_actors[rank]

    def register_serial_func(self):
        for func_name in ["init"]:
            dist_call = partial(self.call_replica_serial_func, func_name)
            setattr(self, func_name, dist_call)

    def register_func(self):
        for func_name in ["model_setup",
                          "before_episode",
                          "after_episode",
                          "validate",
                          "destroy_collective_group",
                          "terminate",
                          "peak_memory",
                          "empty_cache",
                          "set_start_iteration",
                          "offload",
                          "onload",
                          "eval",
                          "train",
                          "set_src_parameter_model",
                          "set_colocate"]:
            dist_call = partial(self.call_replica_func, func_name)
            setattr(self, func_name, dist_call)

    def call_replica_func(self, func, *args, **kwargs):
        refs = []
        for dist_actor in self.replicas:
            ref = getattr(dist_actor, func)(*args, **kwargs)
            if ref is not None:
                refs.append(ref)
        return refs

    def call_replica_serial_func(self, func, *args, **kwargs):
        results = []
        for dist_actor in self.replicas:
            ref = getattr(dist_actor, func)(*args, **kwargs)
            if ref is not None:
                res = future.get(ref)
                results.append(res)
        return results

    def set_colocate_models(self, models):
        self._colocate_models = models

    def colocate_with(self, model):
        return model in self._colocate_models

    @property
    def colocate_models(self):
        return self._colocate_models

    @property
    def all_ranks(self):
        return [dist_actor.all_ranks for dist_actor in self.replicas]

    @property
    def use_vllm_backend(self):
        return vllm_exist and isinstance(self.replicas[0].model, VLLMModule)

    def group_dist_actors_by_tp_rank(self):
        for replica in self.replicas:
            replica.group_dist_actors_by_tp_rank()

    @property
    def enable_offload(self):
        return self.module_args.free_grad_buffers or self.module_args.offload_weights or \
            self.module_args.offload_optimizer_states

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self):
        return f'<{self.__class__.__name__}({self.name}) object at {hex(id(self))}>'
