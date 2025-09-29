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

"""Basic Mapper for Megatron to rollout framework"""
from collections import defaultdict
from typing import List, Dict, TYPE_CHECKING, Union

from megatron.training.utils import unwrap_model

from chatlearn.configs import PolicyConfig
from chatlearn.configs.megatron_config import MegatronPolicyTrainerConfig
from chatlearn.utils.mappings import ShardedTensorInfo

from .mapping_helpers import (
    process_normal_tensor,
    process_gate_up_tensor,
    process_qkv_tensor,
    VLLM_HELPERS,
    HF_HELPERS
)

if TYPE_CHECKING:
    from megatron.core.transformer.module import MegatronModule as MCoreModule
    from chatlearn.models.megatron_module import MegatronModule

class BaseMegatronMapper:
    """BaseMegatronMapper"""
    def __init__(
        self,
        dst_model_config: PolicyConfig,
        model: 'MegatronModule',
        *,
        mapper_config: Union[VLLM_HELPERS, HF_HELPERS] = VLLM_HELPERS,
    ):
        """The Base Mapper for Megatron sync. In each remote Megatron Actor,
        the method of this class is called to generate the parameter mapping
        between src and dst.

        Args:
            dst_model_config (PolicyConfig): The config of target model to
                be sychronized
            model (MegatronModule): The source Megatron Module
            mapper_config (Union[VLLM_HELPERS, HF_HELPERS]): The mapping mode.
        """
        self.model: List['MCoreModule'] = unwrap_model(model.model)
        self._src_model_config: MegatronPolicyTrainerConfig = model.module_args
        self._dst_model_config = dst_model_config
        self._mapper_config = mapper_config
        self._dst_tp_size = 1 if mapper_config.force_full_model else self._dst_model_config.tensor_model_parallel_size
        self._src_name_to_metadata: Dict[str, ShardedTensorInfo] = model.get_parameter_metadata(key_type='local_name')
        self._dst_name_to_metadata: Dict[str, ShardedTensorInfo] = None
        self._mapping = None

    def generate_sync_mapping(
        self,
        dst_name_to_metadata: Dict[str, ShardedTensorInfo]
    ) -> Dict[ShardedTensorInfo, List[ShardedTensorInfo]]:
        """ Generate the synchronization mapping of this local rank.

        Args:
            dst_name_to_metadata (Dict[str, ShardedTensorInfo]): mapping a global
                parameter name to the corresponding ShardedTensorInfo.

        Returns:
            Dict[ShardedTensorInfo, List[ShardedTensorInfo]]: The return
            dict is the plan including all local parameters to be synchronized. The
            mapper will ensure that the key of mapping for each mapping type is
            non-overlapping and can merge into the full state dict of this rank.
            For most cases, the length of dst shards list is 1, except for GQA with
            large TP.
        """
        self._dst_name_to_metadata = dst_name_to_metadata
        return self._map_model()

    def dump_sync_mapping(self, folder_path: str, sync_mapping: Dict):
        """dump the generayed sync mapping to the given folder path in JSON format.
        Currently do nothing.

        Args:
            folder_path (str): The folder path to dump the sync mapping.
            sync_mapping (Dict): The sync mapping to be saved.
        """

    def _map_model(self):
        """Mapping the local name of src model to global name of dst model
        """
        raise NotImplementedError()

    # NOTE: the following function implements the tensor-wise sync mapping
    def _inner_map_for_tensor_parallel(
        self,
        src_key: str,
        dst_key: str,
        *,
        global_expert_id: int=None,
        num_experts: int=None,
        mapping_type: str='column'
    ):
        AXES = {'column': 0, 'row': 1}
        src_info = self._src_name_to_metadata[src_key]
        # NOTE: we should do nothing to bias of RowParallel, call full shape mapping.
        if src_info.ndim == 1 and mapping_type == 'row':
            return self._inner_map_for_full_shape(src_key, dst_key)

        dst_info = self._dst_name_to_metadata[dst_key]
        mapping = {}
        for src_meta, dst_meta in process_normal_tensor(
            src_info,
            self._dst_tp_size,
            axis=AXES[mapping_type]
        ):
            src_meta.param_id, dst_meta.param_id = src_info.param_id, dst_info.param_id
            src_meta.dtype, dst_meta.dtype = src_info.dtype, dst_info.dtype
            if global_expert_id is not None:
                dst_meta = (
                    dst_meta
                    .unsqueeze(offset=global_expert_id, length=num_experts, axis=0)
                    .refragment(1, axis=0) # 1 is dst EP
                )
            mapping[src_meta] = [dst_meta]
        self._update_mapping(mapping)
        return mapping

    def _inner_map_for_full_shape(
        self,
        src_key: str,
        dst_key: str
    ):
        src_info = self._src_name_to_metadata[src_key]
        dst_info = self._dst_name_to_metadata[dst_key]
        results = {src_info.copy(): [dst_info.copy()]}
        self._update_mapping(results)
        return results

    def _inner_map_for_gate_up_proj(self, src_key: str, dst_key: str, proj_type: str, *, global_expert_id: int=None, num_experts: int=None):
        src_info = self._src_name_to_metadata[src_key]
        dst_info = self._dst_name_to_metadata[dst_key]
        mapping = {}
        for src_meta, dst_meta in process_gate_up_tensor(
            src_info,
            self._dst_tp_size,
            proj_type=proj_type
        ):
            src_meta.param_id, dst_meta.param_id = src_info.param_id, dst_info.param_id
            src_meta.dtype, dst_meta.dtype = src_info.dtype, dst_info.dtype
            if global_expert_id is not None:
                dst_meta = (
                    dst_meta
                    .unsqueeze(offset=global_expert_id, length=num_experts, axis=0)
                    .refragment(1, axis=0) # 1 is dst EP
                )
            mapping[src_meta] = [dst_meta]
        self._update_mapping(mapping)
        return mapping

    def _inner_map_for_qkv_proj(self, src_key: str, dst_key: str, proj_type: str, num_attention_heads: int, num_query_groups: int):
        src_info = self._src_name_to_metadata[src_key]
        dst_info = self._dst_name_to_metadata[dst_key]
        mapping = defaultdict(list)
        for src_meta, dst_meta in process_qkv_tensor(
            src_info,
            num_attention_heads,
            num_query_groups,
            self._dst_tp_size,
            proj_type=proj_type
        ):
            src_meta.param_id, dst_meta.param_id = src_info.param_id, dst_info.param_id
            src_meta.dtype, dst_meta.dtype = src_info.dtype, dst_info.dtype
            mapping[src_meta].append(dst_meta)
        self._update_mapping(mapping)
        return mapping

    def _inner_map_for_mla_down_proj(self, src_key: str, dst_key: str):
        src_info = self._src_name_to_metadata[src_key]
        dst_info = self._dst_name_to_metadata[dst_key]
        dst_meta = src_info.refragment(1)
        dst_meta.param_id = dst_info.param_id
        dst_meta.dtype = dst_info.dtype
        results = {src_info.copy(): [dst_meta]}
        self._update_mapping(results)
        return results

    def _update_mapping(self, results: Dict[ShardedTensorInfo, List[ShardedTensorInfo]]) -> None:
        if self._mapping is None:
            self._mapping = defaultdict(list)
        for src_meta, dst_metas in results.items():
            self._mapping[src_meta] += dst_metas
