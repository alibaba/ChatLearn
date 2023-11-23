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
"""model manager"""

from collections import defaultdict

import ray
import ray.experimental.state.api

from chatlearn.data.storage import Storage
from chatlearn.launcher import dlc_utils
from chatlearn.models.torch_module import RLHFTorchModule
from chatlearn.runtime.decorator import decorate_class_func
from chatlearn.runtime.decorator import timeit, preprocess_compute, monitor_error
from chatlearn.runtime.dist_actor import DistActor, DistTorchActor, DistModel
from chatlearn.runtime.parameter_sync import ParameterSyncGroup
from chatlearn.utils.error_monitor import ErrorMonitor, ErrorSignalActor
from chatlearn.utils.logger import logger
from chatlearn.utils.global_vars import set_decorated, is_decorated
from .port_manager import PortManager


class ModelManager:
    """ModelManager"""

    def __init__(self, rlhf_models, resouce_manager, global_args):
        self.local_models = rlhf_models
        self.resouce_manager = resouce_manager
        self.dist_models = []
        self.env_args = global_args.env_args
        self.rlhf_args = global_args.rlhf_args
        self.converted = False
        # port for DLC jobs, the first two ports are reserved for ray start
        self.free_ports = dlc_utils.get_free_ports()[2:]
        self._port_manager = PortManager.remote(self.free_ports)
        self.error_signal = ErrorSignalActor.remote()
        self._storage = Storage.remote()
        self.parameter_sync_groups = {}
        self._parameter_sync_model_mapping = {}
        self.model_packs = []

    def _get_total_device_required(self):
        total_device = 0
        remote_states = set()
        for group in self.rlhf_args.colocation:
            colocate_models = [self._name2distmodel[name] for name in group]
            max_device = max(m.total_device for m in colocate_models)
            total_device += max_device
            for name in group:
                remote_states.add(name)
        for model in self.dist_models:
            # place non-colocate models
            if model.name not in remote_states:
                max_device = model.total_device
                total_device += max_device
        return total_device

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

        total_device_required = self._get_total_device_required()
        if total_device_required > self.resouce_manager.total_gpu:
            raise RuntimeError(f"The number of required gpus for current job is {total_device_required}, " + \
                               f"while the number of applied gpus is {self.resouce_manager.total_gpu}")
        if self.resouce_manager.total_gpu > total_device_required:
            logger.warning(f"The number of applied gpus is {self.resouce_manager.total_gpu}, " + \
                           f"while the number of required gpus is {total_device_required}, " + \
                           f"there is {self.resouce_manager.total_gpu - total_device_required} wasted gpus")

        for group in self.rlhf_args.colocation:
            colocate_models = [self._name2distmodel[name] for name in group]
            self.place_models_to_remote_devices(colocate_models)
            if len(colocate_models) > 1:
                for model in colocate_models:
                    model.need_empty_cache = True
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
            sync_group = ParameterSyncGroup(self._name2distmodel[src_model.name], self._name2distmodel[dst_model.name],
                                            group_name, self.error_signal)
            self.parameter_sync_groups[group_name] = sync_group

    def start_error_monitor(self):
        #group_names = [_ for _ in self.parameter_sync_groups.keys()]
        group_names = list(self.parameter_sync_groups.keys())
        self.error_monitor = ErrorMonitor.remote(self.error_signal, self.dist_models, group_names)
        self.error_monitor.monitor.remote()

    def _get_group_name(self, src_model, dst_model):
        return src_model.name + dst_model.name

    def set_model_sync(self, src_model, tgt_model):
        group_name = self._get_group_name(src_model, tgt_model)
        if group_name in self.parameter_sync_groups:
            logger.warning(f"{group_name} already set, ignore")
        else:
            self._parameter_sync_model_mapping[src_model] = tgt_model

    def sync_parameters(self, requires_grad=None):
        """
        if requires_grad is False, all parameters will be syncronized,
        this happends when broadcast parameters in the beginning of training,
        set the parameters of inference same as training
        """
        for _, sync_group in self.parameter_sync_groups.items():
            sync_group.sync(requires_grad)

    def set_func_decorator(self, model):
        if is_decorated(model.name):
            return

        model_cls = model.__class__
        for func_name in model.call_funcs:
            is_forward_step = func_name == "forward_step"
            decorate_class_func(model_cls, func_name, preprocess_compute, is_forward_step)

        for func_name in ["forward_step", "train_step",
                          "save_checkpoint", "model_setup"]:
            decorate_class_func(model_cls, func_name, timeit, func_name)

        # public user function
        # TODO: use decorator to annotate
        for func_name in ["forward_step", "train_step",
                          "save_checkpoint", "model_setup"]:
            decorate_class_func(model_cls, func_name, monitor_error, func_name)
        set_decorated(model.name)

    def _to_dist_model(self, model):
        """
        Convert one model to DistActor and place it to devices

        Args:
            model: RLHFModule
        """
        self.set_func_decorator(model)
        model.finalize()

        def actor_type():
            if isinstance(model, RLHFTorchModule):
                return DistTorchActor
            else:
                return DistActor

        dist_model = DistModel()
        for replica_id in range(model.num_replica):
            dist_actor = actor_type()(model, self.resouce_manager.gpu_per_node, self.error_signal, self._port_manager,
                                      replica_id, self._storage)
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
        Find model packing strategies that can pack all models into total_device
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
                    if len(unfinished_packs[device]) == 0:
                        unfinished_packs.pop(device)
                    packs.append(model)
                    final_packs.append(packs)
                else:
                    near_devices = [d for d in unfinished_packs if d > device]

                    if near_devices:
                        near_device = sorted(near_devices)[0]
                        packs = unfinished_packs[near_device].pop(0)

                        if len(unfinished_packs[device]) == 0:
                            unfinished_packs.pop(device)
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
        self.model_packs = self.find_model_packing_strategy(models, max_device)

        def _get_model_replica_from_pack(device_index, model_pack):
            device_offset = 0
            for model in model_pack:
                if device_index < device_offset + model.total_device:
                    # compute the model rank
                    model_rank = device_index - device_offset
                    replica_id = model_rank // model.num_device_per_replica
                    return model.replicas[replica_id]
                device_offset += model.total_device
        # 1. we list the models to place on each device
        # 2. for device i, the number of models is N, then the num_gpus for each ray actor is 1.0/N
        device_to_replicas = []
        for i in range(max_device):
            colocate_models = []
            for model_pack in self.model_packs:
                replica = _get_model_replica_from_pack(i, model_pack)
                if replica is not None:
                    colocate_models.append(replica)
            device_to_replicas.append(colocate_models)

        for i, replicas in enumerate(device_to_replicas):
            num_gpus = 1.0 / len(replicas)
            group = i // self.resouce_manager.gpu_per_node
            for replica in replicas:
                replica.create_actor(num_gpus, placement_group, group)

        models_to_revert = self._find_param_recv_models(models)
        for model in models:
            if model in models_to_revert: # pylint: disable=simplifiable-if-statement
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
        for dist_model in self._name2distmodel.values():
            for dist_actor in dist_model.replicas:
                for actor in dist_actor.all_actors:
                    ray.kill(actor)
        ray.kill(self._storage)
        ray.kill(self.error_signal)
        self.resouce_manager.remove_placement_groups()
