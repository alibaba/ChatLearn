import torch
import traceback
import ray
from rlhf.model_wrapper import RLHFModelWrapper, RLHFTorchWrapper
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import inspect
from rlhf.utils import parse_function_args, parse_function_return_num
from functools import partial


RAY_REMOTE = "remote"



def is_tensor(data):
    return isinstance(data, torch.Tensor)


def to_device(device, args):
    if isinstance(args, list):
        args = [to_device(device, arg) for arg in args]
    elif isinstance(args, dict):
        for key, value in args.items():
            args[key] = to_device(device, value)
    elif isinstance(args, torch.Tensor):
        args = args.to(device)
    return args


def device_converter(func):
    """
    convert input to cuda and convert output to cpu
    """
    def inner(self, *args, **kwargs):
        cuda_args = to_device('cuda', args)
        cuda_kwargs = to_device('cuda', kwargs)
        try:
            ret = func(self, *cuda_args, **cuda_kwargs)
            ret = to_device('cpu', ret)
            return ret
        except Exception as e:
            print(f"catch exception ========= in {self.name} {e}, {traceback.format_exc()}", flush=True)
            ray.get(self.error_signal.set.remote())
            raise
    return inner


class DistActor:
    """Manage a collection of actors"""

    def __init__(self, model: RLHFModelWrapper,
                 placement_groups,
                 gpu_per_node,
                 error_signal,
                 port=None):
        self.num_device = model.num_device
        self.gpu_per_process = model.gpu_per_process
        self.placement_groups = placement_groups
        self.model = model
        self.gpu_per_node = gpu_per_node
        self.all_actors = []
        self.port = port
        self.name = self.model.name
        self.error_signal = error_signal
        self.set_device_converter()
        # ranks for model update
        self.all_ranks = None
        self._init_done = False
        self.remote()
        self.rank_to_actors = {}


    def set_device_converter(self):
        for func_name in ["forward_step", "train_step"]:
            model_cls = self.model.__class__
            func = getattr(model_cls, func_name)
            setattr(model_cls, func_name, device_converter(func))


    def _remote_one_model(self, placement_group):
        num_actors = self.num_device // self.gpu_per_process
        dist_actors = []
        
        for i in range(num_actors):
            group = i // self.gpu_per_node
            
            # put actor to corresponding bundle
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=group,
                )
            actor = ray.remote(num_gpus=self.gpu_per_process)(self.model.__class__) \
                       .options(scheduling_strategy=scheduling_strategy) \
                       .remote(self.model.name, self.model.global_args)
            actor.set_error_signal.remote(self.error_signal)
            dist_actors.append(actor)
        return dist_actors


    def remote(self):
        if self._init_done:
            raise RuntimeError("DistActor has been init")
        assert len(self.placement_groups) == self.model.num_replica
        for pg in self.placement_groups:
            self.all_actors.append(self._remote_one_model(pg))
        self.add_remote_func()
        self._init_done = True
        return self

    
    @property
    def actor_num(self):
        return len(self.all_actors) * len(self.all_actors[0])


    def _get_func_args(self, func_name):
        func = getattr(self.model, func_name)
        return parse_function_args(func)

        
    def add_remote_func(self):
        actor = self.all_actors[0][0]

        for func_name, func in inspect.getmembers(actor):
            if func_name.startswith('_'):
                continue
            dist_call = partial(self.call_remote_funcs, func_name)
            setattr(self, func_name, dist_call)


    def call_remote_funcs(self, func_name, *args):
        """
        Call remote functions for a collection of actors.
        """
        results = []
        for replica_id in range(self.model.num_replica):
            for actor in self.all_actors[replica_id]:
                func = getattr(actor, func_name)
                remote_func = getattr(func, RAY_REMOTE)
                res = remote_func(*args)
                results.append(res)
        return results


    def _setup_collective_group(self, rank_offset, world_size, group_name, backend="nccl"):
        refs = []
        i = 0
        all_ranks = []
        for actors in self.all_actors:
            ranks = []
            for actor in actors:
                rank = i+rank_offset
                ref = actor.setup_collective_group.remote(
                    rank=rank,
                    world_size=world_size,
                    backend=backend,
                    group_name=group_name)
                refs.append(ref)
                ranks.append(rank)
                self.rank_to_actors[rank] = actor
                i += 1
            all_ranks.append(ranks)
        self.all_ranks = all_ranks
        return refs


    def get(self, rank):
        # given rank, return the actor
        return self.rank_to_actors[rank]

    
    def terminate(self):
        # terminate when catching exceptions
        for actors in self.all_actors:
            for actor in actors:
                ray.kill(actor)


class DistTorchActor(DistActor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(self.model.num_replica):
            actors = self.all_actors[i]
            self.set_dist_env(actors)
    
    def reorder_actors(self, actors):
        gpu_ids = []
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
        if self.model.global_args.env_args.platform == "DLC":
            master_port = self.port

        world_size = len(actors)
        env_config = {"MASTER_ADDR": master_addr, "MASTER_PORT": master_port, "WORLD_SIZE": world_size}
        ret = []
        for rank, actor in enumerate(actors):
            env_config["RANK"] = rank
            ret.append(actor.set_env.remote(env_config))
        status = sum(ray.get(ret))
        assert status == world_size


    def remote(self):
        super().remote()
        for i in range(self.model.num_replica):
            actors = self.all_actors[i]
            self.set_dist_env(actors)
        return self
