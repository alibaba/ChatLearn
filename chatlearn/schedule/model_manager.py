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
"""model manager"""

import concurrent.futures
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import ray
import ray.experimental.state.api

from chatlearn.data.storage import Storage
from chatlearn.launcher import dlc_utils
from chatlearn.models.torch_module import TorchModule
from chatlearn.runtime.decorator import decorate_class_func
from chatlearn.runtime.decorator import timeit, preprocess_compute, monitor_error
from chatlearn.runtime.dist_actor import DistActor, DistTorchActor, DistModel
from chatlearn.runtime.parameter_sync import ParameterSyncGroup
from chatlearn.utils.error_monitor import ErrorMonitor, ErrorSignalActor
from chatlearn.utils.logger import logger
from chatlearn.utils.global_vars import set_decorated, is_decorated
from .port_manager import PortManager
from ..utils import future


class ModelManager:
    """ModelManager"""

    def __init__(self, models, resouce_manager, global_args):
        self.local_models = models
        self.resouce_manager = resouce_manager
        self.dist_models = []
        self.env_args = global_args.env_args
        self.runtime_args = global_args.runtime_args
        self.converted = False
        # port for DLC jobs, the first two ports are reserved for ray start
        self.free_ports = dlc_utils.get_free_ports()[2:]
        self._port_manager = PortManager.remote(self.free_ports)
        self.error_signal = ErrorSignalActor.remote()
        self._storage = Storage.remote()
        self.parameter_sync_groups = {}
        self._parameter_sync_model_pair = []
        self.model_packs = []
        self.placement_groups = []

    def _get_total_gpu_required(self):
        total_gpu = 0
        remote_states = set()
        for group in self.runtime_args.colocation:
            colocate_models = [self._name2distmodel[name] for name in group]
            max_gpu = max(m.total_gpu for m in colocate_models)
            total_gpu += max_gpu
            for name in group:
                remote_states.add(name)
        for model in self.dist_models:
            # place non-colocate models
            if model.name not in remote_states:
                max_gpu = model.total_gpu
                total_gpu += max_gpu
        return total_gpu

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

        total_gpu_required = self._get_total_gpu_required()
        if total_gpu_required > self.resouce_manager.total_gpu:
            raise RuntimeError(f"The number of required gpus for current job is {total_gpu_required}, " + \
                               f"while the number of applied gpus is {self.resouce_manager.total_gpu}")
        if self.resouce_manager.total_gpu > total_gpu_required:
            logger.warning(f"The number of applied gpus is {self.resouce_manager.total_gpu}, " + \
                           f"while the number of required gpus is {total_gpu_required}, " + \
                           f"there is {self.resouce_manager.total_gpu - total_gpu_required} wasted gpus")

        env_list = []
        for group in self.runtime_args.colocation:
            colocate_models = [self._name2distmodel[name] for name in group]
            self.place_models_to_remote_devices(colocate_models, env_list)
            if len(colocate_models) > 1:
                set_colocate = []
                for model in colocate_models:
                    model.is_colocate = True
                    set_colocate.extend(model.set_colocate(True))
                future.wait(set_colocate)
            for name in group:
                remote_states.add(name)
        for model in self.dist_models:
            # place non-colocate models
            if model.name not in remote_states:
                self.place_models_to_remote_devices([model], env_list)
        self.set_dist_env_concurrent(env_list)
        self.converted = True
        return self.dist_models

    def build_parameter_group(self):
        # set ParameterSyncGroup
        for src_model, dst_model in self._parameter_sync_model_pair:
            group_name = self._get_group_name(src_model, dst_model)
            sync_frequency = self._get_sync_frequency(dst_model)
            sync_group = ParameterSyncGroup(self._name2distmodel[src_model.name], self._name2distmodel[dst_model.name],
                                            group_name, sync_frequency, self.error_signal)
            self.parameter_sync_groups[group_name] = sync_group

    def start_error_monitor(self):
        #group_names = [_ for _ in self.parameter_sync_groups.keys()]
        group_names = list(self.parameter_sync_groups.keys())
        self.error_monitor = ErrorMonitor.remote(self.error_signal, self.dist_models, group_names)
        self.error_monitor.monitor.remote()

    def _get_group_name(self, src_model, dst_model):
        return src_model.name + "2" + dst_model.name

    def _get_sync_frequency(self, model):
        return model.parameter_sync_frequency

    def set_parameter_sync(self, src_model, tgt_model):
        group_name = self._get_group_name(src_model, tgt_model)
        if group_name in self.parameter_sync_groups:
            logger.warning(f"{group_name} already set, ignore")
        else:
            sync_frequency = self._get_sync_frequency(tgt_model)
            assert sync_frequency >= 0, \
                f"parameter sync frequency from {src_model.name} to {tgt_model.name} expected tp be greater than 0, while {sync_frequency}."
            logger.info(f"sync parameters from {src_model.name} to {tgt_model.name} every {sync_frequency} episodes.")
            self._parameter_sync_model_pair.append((src_model, tgt_model))

    def sync_parameters(self, episode_offset=0, requires_grad=None):
        """
        if requires_grad is False, all parameters will be syncronized,
        this happends when broadcast parameters in the beginning of training,
        set the parameters of inference same as training
        """
        for _, sync_group in self.parameter_sync_groups.items():
            if sync_group.frequency and \
                    episode_offset % sync_group.frequency == 0:
                sync_group: ParameterSyncGroup = sync_group

                src_model, dst_model = sync_group.src_model, sync_group.dst_model
                refs = src_model.onload(to_build_grad_buffers=False, to_onload_main_weights=False, to_onload_optimizer_states=False)
                future.wait(refs)
                refs = dst_model.onload(to_build_grad_buffers=False, to_onload_main_weights=False, to_onload_optimizer_states=False)
                future.wait(refs)

                sync_group.sync(requires_grad)

                refs = src_model.offload()
                future.wait(refs)
                refs = dst_model.offload()
                future.wait(refs)

    def set_func_decorator(self, model):
        if is_decorated(model.name):
            return

        model_cls = model.__class__
        for func_name in model.call_funcs:
            is_forward_step = func_name == "forward_step"
            trainable = func_name in model.trainable_funcs
            decorate_class_func(model_cls, func_name, preprocess_compute, is_forward_step, trainable)

        for func_name in ["forward_step", "train_step",
                          "save_checkpoint", "model_setup"]:
            decorate_class_func(model_cls, func_name, timeit, func_name)

        # public user function
        # TODO: use decorator to annotate
        for func_name in ["save_checkpoint", "model_setup", "onload", "offload", "build_dataset",
                          "_build_dataloader", "generate_vllm"] + model.call_funcs:
            decorate_class_func(model_cls, func_name, monitor_error, func_name)
        set_decorated(model.name)

    def _to_dist_model(self, model):
        """
        Convert one model to DistActor and place it to devices

        Args:
            model: BaseModule
        """
        self.set_func_decorator(model)
        model.finalize()

        def actor_type():
            if isinstance(model, TorchModule):
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
            for src, tgt in self._parameter_sync_model_pair:
                if src.name in model_names and model.name == tgt.name:
                    models_to_revert.append(model)
        return models_to_revert

    def find_model_packing_strategy(self, models, total_gpu):
        """
        Find model packing strategies that can pack all models into total_gpu
        try to balance the models among devices, i.e., each device holds similar number of model parts
        e.g., given models A:8, B:4, C:4, total_gpu: 8
        then the pack strategy is [(A), (B,C)]
        """
        sorted_models = sorted(models, key=lambda x: (x.trainable, x.total_gpu), reverse=True)
        assert sorted_models[0].total_gpu <= total_gpu
        final_packs = []
        # key is the remaining gpu
        unfinished_packs = defaultdict(list)

        for model in sorted_models:
            gpu = model.total_gpu
            if gpu == total_gpu:
                final_packs.append([model])
            else:
                if gpu in unfinished_packs:
                    # find a pack
                    packs = unfinished_packs[gpu].pop(0)
                    if len(unfinished_packs[gpu]) == 0:
                        unfinished_packs.pop(gpu)
                    packs.append(model)
                    final_packs.append(packs)
                else:
                    near_gpus = [d for d in unfinished_packs if d > gpu]

                    if near_gpus:
                        near_gpu = sorted(near_gpus)[0]
                        packs = unfinished_packs[near_gpu].pop(0)

                        if len(unfinished_packs[gpu]) == 0:
                            unfinished_packs.pop(gpu)
                        packs.append(model)
                        # update the remaining gpu number
                        unfinished_packs[near_gpu - gpu].append(packs)
                    else:
                        # add model and wait for packing
                        unfinished_packs[total_gpu - gpu].append([model])
        for gpu, packs_list in unfinished_packs.items():
            if packs_list:
                final_packs.extend(packs_list)
        return final_packs

    def place_gpu_models(self, gpu_models, env_list=None):
        if not gpu_models:
            return
        max_gpu = max(m.total_gpu for m in gpu_models)
        placement_group = self.resouce_manager.create_placement_group(max_gpu)
        for i, _ in enumerate(placement_group.bundle_specs):
            self.placement_groups.append((placement_group, i))
        models_str = ','.join([model.name for model in gpu_models])
        logger.info(f"create placement_group {placement_group.bundle_specs} for model {models_str} done")
        for model in gpu_models:
            # TODO: for colocate gpu_per_process > 1, support later
            assert model.gpu_per_process == 1
        self.model_packs = self.find_model_packing_strategy(gpu_models, max_gpu)

        for model in gpu_models:
            pack = []
            for pack in self.model_packs:
                if model in pack:
                    break
            colocate_models = []
            for model2 in gpu_models:
                if model2 is not model and model2 not in pack:
                    colocate_models.append(model2)
            model.set_colocate_models(colocate_models)

        def _get_model_replica_from_pack(gpu_index, model_pack):
            gpu_offset = 0
            for model in model_pack:
                if gpu_index < gpu_offset + model.total_gpu:
                    # compute the model rank
                    model_rank = gpu_index - gpu_offset
                    replica_id = model_rank // model.num_gpu_per_replica
                    return model.replicas[replica_id]
                gpu_offset += model.total_gpu
        # 1. we list the models to place on each device
        # 2. for device i, the number of models is N, then the num_gpus for each ray actor is 1.0/N
        gpu_to_replicas = []
        for i in range(max_gpu):
            colocate_models = []
            for model_pack in self.model_packs:
                replica = _get_model_replica_from_pack(i, model_pack)
                if replica is not None:
                    colocate_models.append(replica)
            gpu_to_replicas.append(colocate_models)

        for i, replicas in enumerate(gpu_to_replicas):
            num_gpus = 1.0 / len(replicas)
            group = i // self.resouce_manager.gpu_per_node
            for replica in replicas:
                replica.create_actor(num_gpus, placement_group, group)

        models_to_revert = self._find_param_recv_models(gpu_models)
        for model in gpu_models:
            if model in models_to_revert: # pylint: disable=simplifiable-if-statement
                # Reverse the placement of tgt models, so that shared models not in the same GPU
                # NCCL limit: NCCL WARN Duplicate GPU detected : rank 1 and rank 0 both on CUDA device
                # TODO: One GPU task still not work
                reverse_gpu_placement = True
            else:
                reverse_gpu_placement = False
            if env_list is None:
                for replica in model.replicas:
                    replica.set_dist_env(reverse_gpu_placement)
            else:
                env_list.append((model, reverse_gpu_placement))

    def place_cpu_models(self, cpu_models):
        if not cpu_models:
            return
        num_cpus = []
        for model in cpu_models:
            for _ in range(model.module_args.num_replica):
                num_cpus.append(model.module_args.cpu_per_process)
        if self.placement_groups is None:
            placement_group = self.resouce_manager.create_placement_group(num_gpus=0, num_cpus=num_cpus, \
                strategy=self.runtime_args.cpu_schedule_strategy)
            models_str = ','.join([model.name for model in cpu_models])
            logger.info(f"create placement_group {placement_group.bundle_specs} for model {models_str} done")
            placement_groups = []
            for i, _ in enumerate(placement_group.bundle_specs):
                placement_groups.append((placement_group, i))
        else:
            placement_groups = self.placement_groups

        i = 0
        for cpu_model in cpu_models:
            for replica in cpu_model.replicas:
                pg, index = placement_groups[i]
                replica.create_actor(0, pg, index)
                i = i + 1
                if i >= len(placement_groups):
                    i = 0

    def place_models_to_remote_devices(self, models, env_list=None):
        cpu_models = [model for model in models if model.total_gpu == 0]
        gpu_models = [model for model in models if model.total_gpu > 0]
        self.place_gpu_models(gpu_models, env_list)
        self.place_cpu_models(cpu_models)
        for model in models:
            for replica in model.replicas:
                replica.preprocess_actors()

    def _set_dist_env(self, model, reverse):
        for replica in model.replicas:
            replica.set_dist_env(reverse)

    def set_dist_env_concurrent(self, env_list):
        num = len(env_list)
        if num == 0:
            return
        with ThreadPoolExecutor(max_workers=num) as executor:
            futures = []
            for model,reverse in env_list:
                # set env
                futures.append(executor.submit(self._set_dist_env, model, reverse))
            for _future in concurrent.futures.as_completed(futures):
                try:
                    _future.result()
                except Exception as e:
                    raise RuntimeError(f"Set dist env generated an exception: {e}") # pylint: disable=raise-missing-from
            concurrent.futures.wait(futures)

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
