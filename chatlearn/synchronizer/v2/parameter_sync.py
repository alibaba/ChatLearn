# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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
import numpy as np
import time
import itertools
import torch

from typing import *

from chatlearn.runtime.dist_actor import DistModel
from chatlearn.launcher.initialize import patch_ray
from chatlearn.utils import future
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.timer import Timers

from chatlearn.utils.mappings import ShardedTensorInfo
from chatlearn.synchronizer.v2.mappers import get_mapper_name

patch_ray()


class ParameterSyncGroup:

    def __init__(self, src_model: DistModel, dst_model: DistModel, group_name: str, frequency, error_signal):
        """Manage Parameter Synchronization between source and destination models.

        Args:
            src_model (DistModel): The source distmodel, only MegatronModel is supported.
            dst_model (DistModel): The destination distmodel, only vLLM backend is supported.
            group_name (str): The tag of this parameter sync group. (Unused)
        """
        self.config = get_args()
        self.src_model, self.dst_model = src_model, dst_model
        self._initialized = False
        # NOTE: for compatability, CURRENTLY NOT USED
        self.frequency = frequency
        self.error_signal = error_signal

    def initialize(self):
        """Intialize the synchronizer. The sync plan is built and validated.
        We assume that the model is static during training process, i.e. no 
        parameter is added or removed, and the datatype or shape of parameters 
        is unchanged. Therefore, the validation is performed only once. 
        Regardless of the above assumption, the initialization can be called 
        multiple times if needed.
        """
        def _initialize_impl(model):
            # TODO: move this code snippset to DistModel
            # NOTE: we define `rank` of each actor as arbitrary unique id.
            actor_id = 0
            for replica in model.replicas:
                replica.all_ranks = []
                for actor in replica.all_actors:
                    replica.rank_to_actors[actor_id] = actor
                    replica.all_ranks.append(actor_id)
                    actor_id += 1
            # TODO: move this code snippset to DistModel
            param_ids = self.generate_global_param_ids(model)
            future.wait(model.call_func_on_all_workers('set_param_ids', param_ids), return_output=True)
            metadata = self.collect_parameter_metadata(model)
            return param_ids, metadata

        self.src_param_ids, self.src_metadatas = _initialize_impl(self.src_model)
        self.dst_param_ids, self.dst_metadatas = _initialize_impl(self.dst_model)
        dst_name_to_metadata = {}
        for name, param_id in self.dst_param_ids.items():
            for rank, metadata_per_rank in self.dst_metadatas.items():
                if param_id in metadata_per_rank:
                    # select one of the TensorMeta for mapper, as mapper don't 
                    # care the shape info of this metadata
                    dst_name_to_metadata[name] = metadata_per_rank[param_id]
                    break

        future.wait(
            self.src_model.call_func_on_all_workers(
                'set_mapper',
                get_mapper_name(self.src_model, self.dst_model), 
                self.dst_model.module_args
            ), 
            return_output=True
        )
        results = future.wait(self.src_model.call_func_on_all_workers(
            'generate_sync_mapping',
            dst_name_to_metadata,
        ), return_output=True)
        global_sync_mapping = self.validate_and_merge_sync_mapping(results)
        breakpoint()
        # planner = get_planner(global_sync_mapping, ...)
        # sync_plan = planner.make_plan()
        # # TODO: Can we find a way to validate plan before actual comm starts?
        # if self.config.parameter_sync.dump_metadata:
        #     # save metadata, sync mapping and plan in readable format
        #     ...
        # # Finally planner will setup synchronizer for src_model and dst_model
        # wait([planner.setup_synchronizer(actor) for actor in ...])
        self._initialized = True

    def collect_parameter_metadata(
        self, 
        model: DistModel
    ) -> Dict[int, Dict[int, ShardedTensorInfo]]:
        """Collect parameter metadata from model. 

        Args:
            model (DistModel): The model to collect parameter metadata.
        
        Returns:
            Dict[int, Dict[int, ShardedTensorInfo]]: The parameter metadata with 
            the following format:
            {
                rank: {
                    param_id: ShardedTensorInfo
                }
            }
        """
        results = future.wait(
            model.call_func_on_all_workers('get_parameter_metadata'), 
            return_output=True
        )
        return {
            rank: res 
            for rank, res in zip(itertools.chain.from_iterable(model.all_ranks), results)
        }

    def generate_global_param_ids(self, model: DistModel) -> Dict[str, int]:
        """This function will generate a global parameter id for each Tensor 
        in the state dict of the model, even if the Tensor will not be synchronized. 
        we also generate a global weight name for each tensor, i.e., weight 
        name w/o model parallel (on elsewhere). Note that model in the same tp group 
        will share the same global weight name/param id.

        Args:
            model (DistModel): The model to generate global parameter ids.
        """
        # Megatron: Dense (TP-CP-DP-PP)  MoE (ETP-EP-EDP-PP)
        # vLLM: Dense (DP-PP-TP), EP group is DP-TP group, i.e., ETP=EDP=1
        results = future.wait(
            model.call_func_on_all_workers('map_local_param_name_to_global'), 
            return_output=True
        )
        param_names = set()
        for res in results:
            param_names.update(res)
        return {name: idx for idx, name in enumerate(param_names)}
    
    def validate_and_merge_sync_mapping(self, sync_mappings: List[Dict]):
        """
            Check whether the merged sync mapping from all source actors meets the metadata
            collected by `collect_parameter_metadata`.
        
        Args:
            sync_mappings (List[Dict]): The sync mappings from all source actors.
        """
        ...
    
    def sync(self, *args, dryrun: bool = False, **kwargs):
        """Intialize and call parameter synchronization on all actors 
        of src and dst models.
        """
        if not self._initialized:
            self.initialize()
        if not dryrun:
            raise NotImplementedError("Currently we only support dryrun in parameter synchronization")
