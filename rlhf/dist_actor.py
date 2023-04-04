import torch
import traceback
import ray
from rlhf.model_wrapper import RLHFModule
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import inspect
from rlhf.utils import parse_function_args, parse_function_return_num
from functools import partial
from rlhf import dlc_utils
from rlhf.logger import logger
from rlhf import utils
import time
import datetime
# TODO: remove this import later
from rlhf.decorator import to_device


RAY_REMOTE = "remote"


class DistActor:
    """Manage a collection of actors"""

    def __init__(self, model: RLHFModule,
                 gpu_per_node, 
                 error_signal,
                 port=None,
                 replica_id=0,
                 storage=None):
        self.num_device = model.num_device
        self.gpu_per_process = model.gpu_per_process
        self.gpu_per_node = gpu_per_node
        self.model = model
        self.all_actors = []
        self.replica_id = replica_id
        self.port = port
        self.name = self.model.name
        self.error_signal = error_signal
        self.storage = storage
        # ranks for model update
        self.all_ranks = None
        self._init_done = False
        self._placement_group = None
        self.rank_to_actors = {}


    @property
    def master(self):
        return self.all_actors[0]


    @property
    def actor_num(self):
        return len(self.all_actors)


    def _get_func_args(self, func_name):
        func = getattr(self.model, func_name)
        return parse_function_args(func)
    

    def preprocess_actors(self):
        self.add_remote_func()

        
    def add_remote_func(self):
        for func_name, func in inspect.getmembers(self.master):
            # ray.actor.ActorMethod
            if func_name.startswith('_'):
                continue
            dist_call = partial(self.call_remote_funcs, func_name)
            setattr(self, func_name, dist_call)


    def call_remote_funcs(self, func_name, *args):
        """
        Call remote functions for a collection of actors.
        """
        results = []
        for actor in self.all_actors:
            func = getattr(actor, func_name)
            remote_func = getattr(func, RAY_REMOTE)
            res = remote_func(*args)
            results.append(res)
        return results


    def create_actor(self, num_gpus, rank, placement_group, group_index):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=group_index,
            )
        actor = ray.remote(num_gpus=num_gpus)(self.model.__class__) \
                   .options(scheduling_strategy=scheduling_strategy) \
                   .remote(self.model.name, self.model.global_args, rank, self.replica_id)
        actor.set_error_signal.remote(self.error_signal)
        actor.set_storage.remote(self.storage)
        self.all_actors.append(actor)


    def _setup_collective_group(self, rank_offset, world_size, group_name, backend="nccl"):
        refs = []
        all_ranks = []
        for i, actor in enumerate(self.all_actors):
            rank = i+rank_offset
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


class DistTorchActor(DistActor):
    
    def reorder_actors(self, actors):
        gpu_per_node = min(self.gpu_per_node, self.model.num_device)
        ordered_actors = []
        count = 0
        actor_gpus = []
        for actor in actors:
            gpus = ray.get(actor.get_visible_gpus.remote())
            count += len(gpus)
            actor_gpus.append((actor, gpus))
            if count == gpu_per_node:
                actor_gpus.sort(key=lambda x: x[1][0])
                ordered_actors += [a[0] for a in actor_gpus]
                actor_gpus = []
                count = 0
        return ordered_actors


    def set_dist_env(self, actors):
        actors = self.reorder_actors(actors)
        master_actor = actors[0]
        master_addr, master_port = ray.get(master_actor.get_addr_port.remote())
        if dlc_utils.in_dlc_env():
            master_port = self.port

        world_size = len(actors)
        env_config = {"MASTER_ADDR": master_addr, "MASTER_PORT": master_port, "WORLD_SIZE": world_size}
        ret = []
        for rank, actor in enumerate(actors):
            env_config["RANK"] = rank
            if self.model.gpu_per_process == 1:
                local_rank = 0
            else:
                local_rank = rank % self.model.gpu_per_process
            env_config["LOCAL_RANK"] = local_rank
            ret.append(actor.set_env.remote(env_config))
        status = sum(ray.get(ret))
        assert status == world_size

    def preprocess_actors(self):
        super().preprocess_actors()
        self.set_dist_env(self.all_actors)
        return self
