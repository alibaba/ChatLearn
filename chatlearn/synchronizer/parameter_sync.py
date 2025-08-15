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
import itertools
from collections import Counter, defaultdict
from typing import List, Dict, TYPE_CHECKING

from chatlearn.utils import future
from chatlearn.utils.mappings import ShardedTensorInfo
from chatlearn.utils.timer import Timers
from chatlearn.utils.logger import logger

from .base_parameter_sync import BaseParameterSyncGroup
from .mappers import get_mapper_name
from .planners import get_planner_cls

if TYPE_CHECKING:
    from chatlearn.runtime.dist_actor import DistModel


class MCoreParameterSyncGroup(BaseParameterSyncGroup):
    """The core implementation of parameter synchronization."""
    def __init__(self, src_model: 'DistModel', dst_model: 'DistModel', frequency: int):
        """Manage Parameter Synchronization between source and destination models.

        Args:
            src_model (DistModel): The source distmodel, only MegatronModel is supported.
            dst_model (DistModel): The destination distmodel, only vLLM backend is supported.
            frequency (int): The synchronization frequency of this group. Should be a positve
            integer.
        """
        super().__init__(src_model, dst_model, frequency)
        self._initialized = False
        # mapping a global weight name to unique id, Dict[str, int]
        self.src_param_ids, self.dst_param_ids = None, None
        # contains the metadata of each rank, Dict[int, List[ShardedTensorInfo]]
        self.src_metadatas, self.dst_metadatas = None, None
        self.plan = None # results of make_plan(), for debugging
        self.timers = Timers()

    def initialize(self):
        """Intialize the synchronizer. The sync plan is built and validated.
        We assume that the model is static during training process, i.e. no
        parameter is added or removed, and the datatype or shape of parameters
        is unchanged. Therefore, the validation is performed only once.
        Regardless of the above assumption, the initialization can be called
        multiple times if needed in the future.
        """
        def _initialize_impl(model):
            param_ids = self.generate_global_param_ids(model)
            future.wait(model.call_func_on_all_workers('set_param_ids', param_ids), return_output=True)
            metadata = self.collect_parameter_metadata(model)
            return param_ids, metadata

        self.timers("metadata").start()
        self.src_param_ids, self.src_metadatas = _initialize_impl(self.src_model)
        self.dst_param_ids, self.dst_metadatas = _initialize_impl(self.dst_model)
        dst_name_to_metadata = {}
        dst_metadatas = {
            rank: {m.param_id: m for m in metadatas} for rank, metadatas in self.dst_metadatas.items()
        }
        for name, param_id in self.dst_param_ids.items():
            for _, metadata_per_rank in dst_metadatas.items():
                if param_id in metadata_per_rank:
                    # select one of the ShardedTensorInfo for mapper, as mapper don't
                    # care the shape info of this metadata
                    dst_name_to_metadata[name] = metadata_per_rank[param_id]
                    break
        self.timers("metadata").stop()

        self.timers("generate-mapping").start()
        future.wait(
            self.src_model.call_func_on_all_workers(
                'set_mapper',
                get_mapper_name(self.src_model, self.dst_model),
                self.dst_model.module_args
            ),
            return_output=True
        )
        results: List[Dict[ShardedTensorInfo, List[ShardedTensorInfo]]] = future.wait(
            self.src_model.call_func_on_all_workers(
                'generate_sync_mapping',
                dst_name_to_metadata,
            ),
            return_output=True
        )
        self.timers("generate-mapping").stop()

        self.timers("validate-mapping").start()
        self.validate_sync_mapping(results)
        self.timers("validate-mapping").stop()

        self.timers("generate-plan").start()
        planner = get_planner_cls(self.src_model, self.dst_model)(
            dict(zip(itertools.chain.from_iterable(self.src_model.all_actor_ids), results)),
            self.dst_metadatas,
        )
        self.plan = planner.make_plan(self.src_model, self.dst_model)
        self.timers("generate-plan").stop()

        # NOTE: Can we find a way to validate plan before actual comm starts?
        self.timers("setup-synchronizer").start()
        planner.setup_synchronizer(self.src_model, self.dst_model, self.plan)
        self.timers("setup-synchronizer").stop()
        self._initialized = True
        logger.info(f"finish parameter sync initialization | {self.timers.log()}")

    def collect_parameter_metadata(
        self,
        model: 'DistModel'
    ) -> Dict[int, List[ShardedTensorInfo]]:
        """Collect parameter metadata from model.

        Args:
            model (DistModel): The model to collect parameter metadata.

        Returns:
            Dict[int, List[ShardedTensorInfo]]: The parameter metadata from 
            each rank.
        """
        results = future.wait(
            model.call_func_on_all_workers('get_parameter_metadata'),
            return_output=True
        )
        results = [list(r.values()) for r in results ]
        return dict(zip(itertools.chain.from_iterable(model.all_actor_ids), results))

    def generate_global_param_ids(self, model: 'DistModel') -> Dict[str, int]:
        """This function will generate a global parameter id for each Tensor
        in the state dict of the model, even if the Tensor will not be synchronized.
        we also generate a global weight name for each tensor, i.e., weight
        name w/o model parallel (on module implementation).

        Args:
            model (DistModel): The model to generate global parameter ids.
        """
        results = future.wait(
            model.call_func_on_all_workers('map_local_param_name_to_global'),
            return_output=True
        )
        param_names = set()
        for res in results:
            param_names.update(res)
        return {name: idx for idx, name in enumerate(param_names)}

    def validate_sync_mapping(
        self,
        sync_mappings: List[Dict[ShardedTensorInfo, List[ShardedTensorInfo]]]
    ):
        """
            Check whether the sync mapping from all source actors 
            meets the dst metadata. 
            
            Note that the sanity of mapping keys should be checked by 
            mapper as SyncGroup doesn't know the detail of the mapping 
            construction, e.g., for GQA mapping between MCore and vLLM, if 
            vLLM TP is larger than `num_query_groups`, KV heads of MCore will
            have more replicas than Q heads.

        Args:
            sync_mappings (List[Dict[ShardedTensorInfo, ShardedTensorInfo]]): The sync 
            mappings from all source actors.
        """
        # 1. each parameter should has K replicas
        counter = Counter()
        for sync_mapping in sync_mappings:
            for infos in sync_mapping.values():
                counter.update(infos)
        param_counter = defaultdict(list)
        for k, v in counter.items():
            param_counter[k.param_id].append(v)

        assert all(max(l) == min(l) for _, l in param_counter.items()), (
            "Each dst parameter should have the same number of replicas"
        )

        # 2. each dst shard should belongs to one actual rank
        metadata_group_by_param_id = defaultdict(set)
        for metadata_per_rank in self.dst_metadatas.values():
            for info in metadata_per_rank:
                metadata_group_by_param_id[info.param_id].add(info)

        for shard in counter:
            flag = False
            target_shards = metadata_group_by_param_id[shard.param_id]
            for info in target_shards:
                flag |= shard in info
                if flag:
                    break
            if not flag:
                raise ValueError(
                    f"{shard} does not belong to any dst metadata, target: {target_shards}"
                )

        # TODO: 3. all unique parameter shards should compose an entire model

    def sync(self, dryrun: bool = False):
        """Perform parameter synchronization on this group. If `dryrun` is True,
        some initialization will be excuted and no actual synchronization 
        will be done.

        Args:
            dryrun (bool, optional): Whether to run in dryrun mode. 
            Defaults to False.
        """
        if not self._initialized:
            self.initialize()
        if dryrun:
            return
        self.timers("communication").start()
        refs = (
            self.src_model.call_func_on_all_workers('parameter_sync') +
            self.dst_model.call_func_on_all_workers('parameter_sync')
        )
        future.wait(refs, return_output=True)
        refs = (
            self.src_model.call_func_on_all_workers('post_parameter_sync') +
            self.dst_model.call_func_on_all_workers('post_parameter_sync')
        )
        future.wait(refs, return_output=True)
        self.timers("communication").stop()
        logger.info(f"finish parameter synchronization | {self.timers.log(names=['communication'])}")
