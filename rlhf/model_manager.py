from rlhf.dist_actor import DistActor, DistTorchActor
from rlhf.model_wrapper import RLHFTorchModule
from rlhf.parameter_sync import ParameterSyncGroup
from rlhf import dlc_utils
from functools import partial
from rlhf.logger import logger
from rlhf.storage import Storage
import ray
import time
import ray.util.collective as col
from collections import defaultdict


class ModelManager:

    def __init__(self, rlhf_models, resouce_manager, global_args):
        self.local_models = rlhf_models
        self.resouce_manager = resouce_manager
        self.remote_models = []
        self.env_args = global_args.env_args
        self.rlhf_args = global_args.rlhf_args
        self.converted = False
        self.free_ports = []
        if dlc_utils.in_dlc_env():
            # port for DLC jobs, the first port is reserved for ray start
            self.free_ports = dlc_utils.get_free_ports()[1:]
        self.port_index = 0
        self.error_signal = ErrorSignalActor.remote()
        self._storage = Storage.remote()
        self.parameter_sync_groups = {}
        self.remote()

    
    def remote(self) -> list:
        """
        convert model to remote
        """
        if self.converted:
            return self.remote_models

        name2model = {}
        remote_states = set()
        for model in self.local_models:
            remote_model = self._to_dist_model(model)
            self.remote_models.append(remote_model)
            name2model[model.name] = remote_model
        
        for group in self.rlhf_args.colocation:
            colocate_models = [name2model[name] for name in group]
            self.place_models_to_remote_devices(colocate_models)
            for name in group:
                remote_states.add(name)
        for model in self.remote_models:
            if model.name not in remote_states:
                self.place_models_to_remote_devices([model])

        self.converted = True
        return self.remote_models


    def start_error_monitor(self):
        group_names = [_ for _ in self.parameter_sync_groups.keys()]
        self.error_monitor = ErrorMonitor.remote(self.error_signal, self.remote_models, group_names)
        self.error_monitor.monitor.remote()


    def set_model_sync(self, src_model, tgt_model):
        group_name = ""
        for model in [src_model, tgt_model]:
            tag = model.name
            group_name += tag
        if group_name in self.parameter_sync_groups:
            logger.warn(f"{group_name} already set, ignore")
        else:
            sync_group = ParameterSyncGroup(src_model, tgt_model, group_name)
            self.parameter_sync_groups[group_name] = sync_group


    def sync_parameters(self):
        for group_name, sync_group in self.parameter_sync_groups.items():
            sync_group.sync()


    def get_free_port(self):
        port = self.free_ports[self.port_index]
        self.port_index += 1
        return port


    def _to_dist_model(self, model):
        """
        Convert one model to DistActor and place it to devices

        Args:
            model: RLHFModule
        """
        def actor_type():
            if isinstance(model, RLHFTorchModule):
                return DistTorchActor
            else:
                return DistActor

        dist_model = DistModel()
        for replica_id in range(model.num_replica):
            free_port = None
            if isinstance(model, RLHFTorchModule):
                if dlc_utils.in_dlc_env():
                    free_port = self.get_free_port()
            dist_actor = actor_type()(model, self.error_signal, free_port, replica_id, self._storage)
            dist_model.add_replica(dist_actor)
        return dist_model
    

    def place_models_to_remote_devices(self, models):
        """
        Args:
            models: a list of DistModel
        """
        num_replica = len(models[0].replicas)
        for model in models:
            # TODO: relax this constraints later
            assert num_replica == len(model.replicas)
        for replica_id in range(num_replica):
            self.place_replica_to_remote_devices([m.replicas[replica_id] for m in models])


    def place_replica_to_remote_devices(self, models):
        """
        Args:
            models: a list of DistActor
        """
        num_device = models[0].num_device
        gpu_per_process = models[0].gpu_per_process
        gpu_per_node = self.resouce_manager.gpu_per_node
        # create placement_group for colocation models
        placement_group = self.resouce_manager.create_placement_group(models[0])

        if len(models) > 1:
            for model in models:
                assert num_device == model.num_device
                assert gpu_per_process == model.gpu_per_process

        num_actors = num_device // gpu_per_process
        num_actors = max(num_actors, 1)
        num_gpus = gpu_per_process / len(models)
        
        for i in range(num_actors):
            group = i // gpu_per_node
            for model in models:
                # put actor to corresponding bundle
                model.create_actor(num_gpus, i, placement_group, group)
        for model in models:
            model.preprocess_actors()


    def clean(self):
        for group in self.parameter_sync_groups.values():
            group.destroy_collective_group()


@ray.remote
class ErrorMonitor:
    def __init__(self, error_signal, remote_models, group_names):
        self.error_signal = error_signal
        self.remote_models = remote_models
        self.collective_groups = group_names



    def monitor(self):
        while True:
            catch_err = ray.get(self.error_signal.is_set.remote())
            if catch_err:
                break
            time.sleep(2)
        logger.exception("error found")
        for group_name in self.collective_groups:
            col.destroy_collective_group(group_name)
        for model in self.remote_models:
            model.terminate()
        try:
            exit_actor = ray.get_actor("ExitActor")
            ray.kill(exit_actor)
        except Exception as e:
            pass
        ray.shutdown()


@ray.remote(num_cpus=0)
class ErrorSignalActor:
    def __init__(self):
        self.error_state = False

    def set(self):
        self.error_state = True

    def is_set(self):
        return self.error_state


class DistModel:

    def __init__(self):
        self.replicas = []
        self.name = None
        self.rank_to_actors = {}
        self.register_func()


    def add_replica(self, replica):
        self.replicas.append(replica)
        self.name = replica.name


    @property
    def actor_num(self):
        return sum([len(dist_actor.all_actors) for dist_actor in self.replicas])


    def get_actor(self, rank):
        # given rank, return the actor
        for dist_actor in self.replicas:
            if rank in dist_actor.rank_to_actors:
                return dist_actor.rank_to_actors[rank]

    def register_func(self):
        for func_name in ["setup",
                          "before_episode",
                          "after_episode",
                          "validate",
                          "destroy_collective_group",
                          "terminate",
                          "init",
                          "peak_memory"]:
            dist_call = partial(self.call_replica_func, func_name)
            setattr(self, func_name, dist_call)


    def call_replica_func(self, func, *args, **kwargs):
        refs = []
        for dist_actor in self.replicas:
            ref = getattr(dist_actor, func)(*args, **kwargs)
            if ref is not None:
                refs.append(ref)
        return refs

    
    @property
    def all_ranks(self):
        return [dist_actor.all_ranks for dist_actor in self.replicas]
