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
from rlhf.decorator import timeit, preprocess_compute, monitor_error
from rlhf.decorator import decorate_class_func
from collections import defaultdict


class ModelManager:

    def __init__(self, rlhf_models, resouce_manager, global_args):
        self.local_models = rlhf_models
        self.resouce_manager = resouce_manager
        self.dist_models = []
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
        self._parameter_sync_model_mapping = {}

    
    def remote(self) -> list:
        """
        convert model to remote
        """
        if self.converted:
            return self.dist_models

        self._name2distmodel = {}
        remote_states = set()
        for model in self.local_models:
            dist_model = self._to_dist_model(model)
            self.dist_models.append(dist_model)
            self._name2distmodel[model.name] = dist_model

        for group in self.rlhf_args.colocation:
            colocate_models = [self._name2distmodel[name] for name in group]
            self.place_models_to_remote_devices(colocate_models)
            for name in group:
                remote_states.add(name)
        for model in self.dist_models:
            # place non-colocate models
            if model.name not in remote_states:
                self.place_models_to_remote_devices([model])
        self.converted = True
        return self.dist_models


    def build_parameter_group(self):
        # set ParameterSyncGroup
        for src_model, dst_model in self._parameter_sync_model_mapping.items():
            group_name = self._get_group_name(src_model, dst_model)
            sync_group = ParameterSyncGroup(self._name2distmodel[src_model.name], self._name2distmodel[dst_model.name], group_name)
            self.parameter_sync_groups[group_name] = sync_group


    def start_error_monitor(self):
        group_names = [_ for _ in self.parameter_sync_groups.keys()]
        self.error_monitor = ErrorMonitor.remote(self.error_signal, self.dist_models, group_names)
        self.error_monitor.monitor.remote()

    def _get_group_name(self, src_model, dst_model):
        return src_model.name + dst_model.name


    def set_model_sync(self, src_model, tgt_model):
        group_name = self._get_group_name(src_model, tgt_model)
        if group_name in self.parameter_sync_groups:
            logger.warn(f"{group_name} already set, ignore")
        else:
            self._parameter_sync_model_mapping[src_model] = tgt_model


    def sync_parameters(self):
        for group_name, sync_group in self.parameter_sync_groups.items():
            sync_group.sync()


    def get_free_port(self):
        port = self.free_ports[self.port_index]
        self.port_index += 1
        return port

    def set_func_decorator(self, model):
        model_cls = model.__class__
        for func_name in model.call_funcs:
            merge_input = func_name == "forward_step"
            decorate_class_func(model_cls, func_name, preprocess_compute, merge_input)

        for func_name in ["forward_step", "train_step",
                          "save_checkpoint", "setup"]:
            decorate_class_func(model_cls, func_name, timeit, func_name)

        # public user function
        # TODO: use decorator to annotate
        for func_name in ["forward_step", "train_step",
                          "save_checkpoint", "setup"]:
            decorate_class_func(model_cls, func_name, monitor_error, func_name)


    def _to_dist_model(self, model):
        """
        Convert one model to DistActor and place it to devices

        Args:
            model: RLHFModule
        """
        self.set_func_decorator(model)
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
            dist_actor = actor_type()(model, self.resouce_manager.gpu_per_node, self.error_signal, free_port, replica_id, self._storage)
            dist_model.add_replica(dist_actor)
        return dist_model


    def _find_param_recv_models(self, models):
        """
        find models that recv parameters
        """
        if len(models) < 2:
            return []
        model_names = [model.name for model in models]
        models_to_revert = []
        for model in models:
            for src, tgt in self._parameter_sync_model_mapping.items():
                if src.name in model_names and model.name == tgt.name:
                    models_to_revert.append(model)
        return models_to_revert


    def find_model_packing_strategy(self, models, total_device):
        """
        Find model packing strategies that can pack all models into num_device
        try to balance the models among devices, i.e., each device holds similar number of model parts
        e.g., given models A:8, B:4, C:4, total_device: 8
        then the pack strategy is [(A), (B,C)]
        """
        sorted_models = sorted(models, key=lambda x: x.total_device, reverse=True)
        assert sorted_models[0].total_device <= total_device
        final_packs = []
        # key is the remaining device
        unfinished_packs = defaultdict(list)
        for model in sorted_models:
            device = model.total_device
            if device == total_device:
                final_packs.append([model])
            else:
                if device in unfinished_packs:
                    # find a pack
                    packs = unfinished_packs[device].pop(0)
                    packs.append(model)
                    final_packs.append(packs)
                else:
                    near_devices = [d for d in unfinished_packs if d > device]

                    if near_devices:
                        near_device = sorted(near_devices)[0]
                        packs = unfinished_packs[near_device].pop(0)
                        packs.append(model)
                        # update the remaining device number
                        unfinished_packs[near_device - device].append(packs)
                    else:
                        # add model and wait for packing
                        unfinished_packs[total_device - device].append([model])
        for device, packs_list in unfinished_packs.items():
            if packs_list:
                final_packs.extend(packs_list)
        return final_packs



    def place_models_to_remote_devices(self, models):
        max_device = max(m.total_device for m in models)
        placement_group = self.resouce_manager.create_placement_group(max_device)
        logger.info(f"create placement_group {placement_group.bundle_specs} for model {models} done")
        if len(models) > 1:
            for model in models:
                # TODO: for colocate gpu_per_process > 1, support later
                assert model.gpu_per_process == 1
        model_packs = self.find_model_packing_strategy(models, max_device)
        num_gpus = 1.0 / len(model_packs)

        def _get_model_replica_from_pack(device_index, model_pack):
            device_offset = 0
            for model in model_pack:
                if device_index < device_offset + model.total_device:
                    # compute the model rank
                    model_rank = device_index - device_offset
                    replica_id = model_rank // model.num_device_per_replica
                    return model.replicas[replica_id]
                device_offset += model.total_device

        # for model_pack in model_packs:
        for i in range(max_device):
            group = i // self.resouce_manager.gpu_per_node
            for model_pack in model_packs:
                replica = _get_model_replica_from_pack(i, model_pack)
                replica.create_actor(num_gpus, placement_group, group)
        models_to_revert = self._find_param_recv_models(models)
        for model in models:
            if model in models_to_revert:
                # Reverse the placement of tgt models, so that shared models not in the same GPU
                # NCCL limit: NCCL WARN Duplicate GPU detected : rank 1 and rank 0 both on CUDA device
                # TODO: One GPU task still not work
                reverse_device_placement = True
            else:
                reverse_device_placement = False
            for replica in model.replicas:
                replica.preprocess_actors(reverse_device_placement)


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
        self.register_serial_func()
        self.register_func()


    def add_replica(self, replica):
        self.replicas.append(replica)
        self.name = replica.name


    @property
    def module_args(self):
        return self.replicas[0].module_args



    @property
    def actor_num(self):
        return sum([len(dist_actor.all_actors) for dist_actor in self.replicas])


    @property
    def num_replica(self):
        return len(self.replicas)


    @property
    def total_device(self):
        return self.num_replica * self.replicas[0].num_device


    @property
    def num_device_per_replica(self):
        return self.replicas[0].num_device

    @property
    def gpu_per_process(self):
        return self.replicas[0].gpu_per_process


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
        for func_name in ["setup",
                          "before_episode",
                          "after_episode",
                          "validate",
                          "destroy_collective_group",
                          "terminate",
                          "peak_memory",
                          "empty_cache"]:
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
                res = utils.get(ref)
                results.append(res)
        return results

    
    @property
    def all_ranks(self):
        return [dist_actor.all_ranks for dist_actor in self.replicas]


    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self):
        return f'<{self.__class__.__name__}({self.name}) object at {hex(id(self))}>'
