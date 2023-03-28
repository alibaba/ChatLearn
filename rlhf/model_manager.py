from rlhf.dist_actor import DistActor, DistTorchActor
from rlhf.model_wrapper import RLHFTorchModule
from rlhf.parameter_sync import ParameterSyncGroup
from rlhf import dlc_utils
from functools import partial
from rlhf.logger import logger
from rlhf.storage import Storage
from rlhf import utils
import ray
import time
import ray.util.collective as col


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
        for model in self.local_models:
            self.remote_models.append(self._to_dist_actor(model))
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


    def _to_dist_actor(self, model) -> DistActor:
        """
        Convert one model to DistActor and place it to devices
        """
        def actor_type():
            if isinstance(model, RLHFTorchModule):
                return DistTorchActor
            else:
                return DistActor

        dist_model = DistModel()
        for replica_id in range(model.num_replica):
            placement_group = self.resouce_manager.get_placement_group(model, replica_id)
            gpu_per_node = self.resouce_manager.gpu_per_node
            free_port = None
            if isinstance(model, RLHFTorchModule):
                if dlc_utils.in_dlc_env():
                    free_port = self.get_free_port()
            dist_actor = actor_type()(model, placement_group, gpu_per_node, self.error_signal, free_port, replica_id, self._storage)
            dist_model.add_replica(dist_actor)
        return dist_model


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
                          "compile_dependencies",
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
