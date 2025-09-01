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
                 replica_id=0):
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
        # ranks for model update
        self.all_actor_ids = None
        self._init_done = False
        self._placement_group = None
        self.id_to_actors = {}

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

    def _create_actor(self, cls, num_gpus, placement_group, group_index, **kwargs):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=group_index,
        )
        # use max_concurrency=1 to make sure only one task execute at one time
        actor = ray.remote(num_gpus=num_gpus, num_cpus=0)(cls) \
            .options(scheduling_strategy=scheduling_strategy) \
            .remote(self.model.name, self.model.global_args, self.replica_id, **kwargs)
        actor.set_error_signal.remote(self.error_signal)
        self.all_actors.append(actor)
        return actor

    def create_actor(self, num_gpus, placement_group, group_index):
        return self._create_actor(self.model.__class__, num_gpus, placement_group, group_index)

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

    def group_dist_actors_by_dp_rank(self):
        """
        get a dp rank to actors map(self.dp_rank_to_actors).
        Warning: vllm DistActor may have an additional engine
        """
        self.dp_rank_to_actors = defaultdict(list)
        self.data_parallel_size = future.get(self.all_actors[0].get_data_parallel_size.remote())
        if self.data_parallel_size is None:
            self.data_parallel_size = 1
        dp_ranks = future.wait([actor.get_data_parallel_rank.remote() for actor in self.all_actors], return_output=True)
        for actor, dp_rank in zip(self.all_actors, dp_ranks):
            self.dp_rank_to_actors[dp_rank].append(actor)

    def set_dist_env(self, revert_placement=False):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})[{self.replica_id}]"

    def __repr__(self):
        return f'<{self.__class__.__name__}({self.name})[{self.replica_id}] object at {hex(id(self))}>'


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

    def set_dist_env(self, revert_placement=False, extra_env=None):
        self.all_actors = self.reorder_actors(self.all_actors, revert_placement)
        master_addr = future.get(self.master.get_address.remote())
        master_port = future.get(self._port_manager.get_free_port.remote(master_addr))

        world_size = self.actor_num
        env_config = {"MASTER_ADDR": master_addr, "MASTER_PORT": master_port, "WORLD_SIZE": world_size}
        if extra_env:
            env_config.update(extra_env)
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

class DistVLLMActor(DistTorchActor):
    """DistVLLMActor"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = None

    def create_actor(self, num_gpus, placement_group, group_index):

        kwargs = {
            "vllm_actor_type" : "worker",
            "rpc_rank" : len(self.all_actors)
        }

        self._create_actor(self.model.__class__, num_gpus, placement_group, group_index, **kwargs)

    def create_engine_actor(self, num_gpus, placement_group, group_index):
        self.engine = self._create_actor(self.model.__class__, num_gpus, placement_group, group_index)

    def call_vllm_engine_remote_funcs(self, func_name, *args, **kwargs):
        """
        Call remote functions for vllm_engine.
        """
        results = []
        res = self.call_actor_remote_func(self.engine, func_name, *args, **kwargs)
        results.append(res)
        return results

    def call_vllm_engine_and_workers_remote_funcs(self, func_name, *args, **kwargs):
        """
        Call remote functions for vllm_engine + workers.
        """
        results = []
        for actor in self.all_actors:
            res = self.call_actor_remote_func(actor, func_name, *args, **kwargs)
            results.append(res)
        res = self.call_actor_remote_func(self.engine, func_name, *args, **kwargs)
        results.append(res)
        return results

    def add_remote_func(self):
        for func_name, _ in inspect.getmembers(self.master):
            if func_name.startswith("_"):
                continue
            if func_name in [
                "eval_forward",
                "forward_step",
                "setup_engine",
                "generate_vllm",
                "offload",
                "onload",
                "get_and_clear_metrics",
                "timer_summary",
            ]:
                dist_call = partial(self.call_vllm_engine_remote_funcs, func_name)
            elif func_name in ["model_setup"]:
                dist_call = partial(
                    self.call_vllm_engine_and_workers_remote_funcs, func_name
                )
            else:
                dist_call = partial(self.call_remote_funcs, func_name)
            setattr(self, func_name, dist_call)

    def setup_vllm_engine(self):
        return self.engine.setup_engine.remote(self.all_actors)

    @property
    def master(self):
        return self.engine

    def peak_memory(self):
        return self.model.peak_memory()

class DistSGLangActor(DistTorchActor):
    """DistSGLangActor"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = None

    def create_actor(self, num_gpus, placement_group, group_index):

        self._create_actor(self.model.__class__, num_gpus, placement_group, group_index)

        self.engine = self.all_actors[0]

    def call_engine_remote_funcs(self, func_name, *args, **kwargs):
        """
        Call remote functions for engine.
        """
        results = []
        res = self.call_actor_remote_func(self.engine, func_name, *args, **kwargs)
        results.append(res)
        return results

    @property
    def master(self):
        return self.engine

    def add_remote_func(self):
        for func_name, _ in inspect.getmembers(self.all_actors[0]):
            if func_name.startswith("_"):
                continue
            if func_name in [
                "eval_forward",
                "forward_step",
                "generate",
                "flush_cache",
                "offload",
                "onload",
                "get_and_clear_metrics",
                "timer_summary",
            ]:
                dist_call = partial(self.call_engine_remote_funcs, func_name)
            else:
                dist_call = partial(self.call_remote_funcs, func_name)
            setattr(self, func_name, dist_call)

    def set_dist_env(self, revert_placement=False, extra_env=None):
        master_addr = future.get(self.master.get_address.remote())
        sgalng_nccl_port = future.get(self._port_manager.get_free_port.remote(master_addr))
        extra_env = {"SGLANG_NCCL_PORT": sgalng_nccl_port}
        super().set_dist_env(revert_placement=revert_placement, extra_env=extra_env)


class DistModel:
    """DistModel"""

    def __init__(self):
        self.replicas = []
        self.name = None
        self.id_to_actors = {}
        self.register_func()
        self._is_colocate = False
        self._colocate_models = []

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
    def runtime_args(self):
        return self.replicas[0].runtime_args

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
            if rank in dist_actor.id_to_actors:
                return dist_actor.id_to_actors[rank]

    def init(self):
        """initialize on all actors of this DistModel and
        assign a unique id to each actor. We ensure id
        """
        refs = []
        for dist_actor in self.replicas:
            # TODO: actually call basemodule.init() on all workers?
            refs.append(dist_actor.init())
        future.get(refs)

        # NOTE: we ensure the id starts from 0
        actor_id = 0
        for replica in self.replicas:
            replica.all_actor_ids = []
            for actor in replica.all_actors:
                replica.id_to_actors[actor_id] = actor
                replica.all_actor_ids.append(actor_id)
                actor_id += 1

    def register_func(self):
        for func_name in ["model_setup",
                          "before_episode",
                          "after_episode",
                          "get_and_clear_metrics",
                          "validate",
                          "terminate",
                          "peak_memory",
                          "empty_cache",
                          "set_start_iteration",
                          "offload",
                          "onload",
                          "eval",
                          "train",
                          "set_colocate",
                          "setup_engine",
                          "timer_summary"]:
            dist_call = partial(self.call_replica_func, func_name)
            setattr(self, func_name, dist_call)

    def call_func_on_all_workers(self, func_name: str, *args, **kwargs):
        """Call some worker function on all workers of this 
        DistModel.

        NOTE: engine of vLLM is never called by this function.

        Args:
            func_name (str): the function name to be called
        """
        refs = []
        for dist_actor in self.replicas:
            refs.extend(dist_actor.call_remote_funcs(func_name, *args, **kwargs))
        return refs

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
    def all_actor_ids(self):
        return [dist_actor.all_actor_ids for dist_actor in self.replicas]

    @property
    def use_vllm_backend(self):
        return vllm_exist and isinstance(self.replicas[0].model, VLLMModule)

    def group_dist_actors_by_dp_rank(self):
        for replica in self.replicas:
            replica.group_dist_actors_by_dp_rank()

    @property
    def enable_offload(self):
        return self.module_args.free_gpu_memory.free_grad_buffers or self.module_args.free_gpu_memory.offload_weights or \
            self.module_args.free_gpu_memory.offload_optimizer_states

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self):
        return f'<{self.__class__.__name__}({self.name}) object at {hex(id(self))}>'
