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
"""Mapper for Megatron to vLLM"""
from collections import defaultdict
from typing import List, Dict, Tuple, TYPE_CHECKING

import torch
from torch import nn

from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.training.utils import unwrap_model

from chatlearn.configs.common import PolicyConfig
from chatlearn.configs.megatron_config import MegatronPolicyTrainerConfig
from chatlearn.utils.mappings import ShardedTensorInfo

from .mapping_helpers import (
    process_normal_tensor,
    process_gate_up_tensor,
    process_qkv_tensor
)

if TYPE_CHECKING:
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    from megatron.core.transformer.transformer_layer import TransformerLayer
    from megatron.core.tensor_parallel import ColumnParallelLinear
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention
    from megatron.core.transformer.attention import SelfAttention

    from chatlearn.models.megatron_module import MegatronModule


class MegatronVLLMMapper:
    """MegatronVLLMMapper"""
    def __init__(
        self,
        dst_model_config: PolicyConfig,
        model: 'MegatronModule',
    ):
        """The Mapper for Megatron to vLLM sync. In each remote Megatron Actor,
        the method of this class is called to generate the parameter mapping
        between src and dst.

        WARNING: Currently, the mapper assumes that the weights name of same
        submodules in different vLLM models are still same.

        Args:
            dst_model_config (PolicyConfig): The config of target model to
                be sychronized
            model (MegatronModule): The source Megatron Module
        """
        # NOTE: the model in one megatron rank is always a list of GPTModel
        # (length > 1 when VPP is enabled).
        self.model: List['GPTModel'] = [unwrap_model(model.megatron_model())]
        if len(self.model) != 1 or getattr(self.model[0], 'vp_stage', None) is not None:
            raise NotImplementedError("Currently, the mapper does not support VPP")
        self._src_model_config: MegatronPolicyTrainerConfig = model.module_args
        self._dst_model_config = dst_model_config
        assert self._dst_model_config.expert_model_parallel_size == 1, "Currently EP > 1 is not supported"
        self._src_name_to_param_ids = model.local_name_to_param_id
        self._src_name_to_metadata = model.get_parameter_metadata(key_type='local_name')

    def generate_sync_mapping(
        self,
        dst_name_to_metadata: Dict[str, Tuple[int, torch.dtype]]
    ) -> Dict[ShardedTensorInfo, List[ShardedTensorInfo]]:
        """ Generate the synchronization mapping of this local rank.

        Args:
            dst_name_to_metadata (Dict[str, Tuple[int, torch.dtype]]): mapping a global
                parameter name to its param_id and datatype

        Returns:
            Dict[ShardedTensorInfo, List[ShardedTensorInfo]]: The return
            dict is the plan including all local parameters to be synchronized. The
            mapper will ensure that the key of mapping for each mapping type is
            non-overlapping and can merge into the full state dict of this rank.
            For most cases, the length of dst shards list is 1, except for GQA with 
            large vLLM TP.
        """
        self._dst_name_to_metadata = dst_name_to_metadata
        return self._map_model()

    def dump_sync_mapping(self, folder_path: str, sync_mapping: Dict):
        """dump the generayed sync mapping to the given folder path in JSON format.

        Args:
            folder_path (str): The folder path to dump the sync mapping.
            sync_mapping (Dict): The sync mapping to be saved.
        """
        raise NotImplementedError()

    # NOTE: the following function implements the module-wise sync mapping
    def _map_model(self):
        """Mapping the local name of src model to global name of
        dst model
        """
        mapping = {}
        for vpp_index, model in enumerate(self.model):
            if len(self.model) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_index)

            layer_offset = get_transformer_layer_offset(model.config)
            if model.mtp_process:
                raise NotImplementedError("Currently, the mapper does not support MTP")
            if model.pre_process:
                mapping.update(self._map_preprocess_layer(
                    model.embedding,
                    src_prefix="embedding.",
                    dst_prefix="model.",
                ))
            for layer_idx in range(model.decoder.num_layers_per_pipeline_rank):
                global_layer_id = layer_offset + layer_idx
                mapping.update(self._map_decoder_layer(
                    model.decoder.layers[layer_idx],
                    src_prefix=f"decoder.layers.{layer_idx}.",
                    dst_prefix=f"model.layers.{global_layer_id}.",
                ))
            if model.post_process:
                mapping.update(self._map_norm_layer(
                    model.decoder.final_layernorm,
                    src_prefix="decoder.final_layernorm.",
                    dst_prefix="model.norm.",
                ))
                if not model.share_embeddings_and_output_weights:
                    mapping.update(self._map_postprocess_layer(
                        model.output_layer,
                        src_prefix="output_layer.",
                        dst_prefix="",
                    ))

            if len(self.model) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(None)
        return mapping

    def _map_norm_layer(self, module: nn.Module, src_prefix: str='', dst_prefix: str='', *, is_norm_layer: bool=True):
        """If is_norm_layer is True, try to map on all possible keys,
        otherwise only map on `layer_norm_weight` and `layer_norm_bias`
        """
        mapping = {}
        _keynames = {
            'weight': 'weight',
            'bias': 'bias',
            'layer_norm_weight': 'weight',
            'layer_norm_bias': 'bias'
        }
        possible_keys = ['layer_norm_weight', 'layer_norm_bias']
        if is_norm_layer:
            possible_keys += ['weight', 'bias']
        for item in possible_keys:
            if getattr(module, item, None) is None or getattr(module, item).numel() == 0:
                continue
            mapping.update(self._inner_map_for_full_shape(
                f"{src_prefix}{item}",
                f"{dst_prefix}{_keynames[item]}"
            ))
        return mapping

    def _map_decoder_layer(self, module: 'TransformerLayer', src_prefix: str='', dst_prefix: str=''):
        mapping = {}
        if self._src_arch.multi_latent_attention:
            map_attn_func = self._map_mla_selfattn
            norm_layer = module.input_layernorm
            norm_src_key = f"{src_prefix}input_layernorm."
            norm_dst_key = f"{dst_prefix}input_layernorm."
            is_norm_layer = True
        else:
            map_attn_func = self._map_selfattn
            norm_layer = module.self_attention.linear_qkv
            norm_src_key = f"{src_prefix}self_attention.linear_qkv."
            norm_dst_key = f"{dst_prefix}input_layernorm."
            is_norm_layer = False
        mapping.update(map_attn_func(module.self_attention, src_prefix=f"{src_prefix}self_attention.", dst_prefix=f"{dst_prefix}self_attn."))
        mapping.update(self._map_norm_layer(norm_layer, norm_src_key, norm_dst_key, is_norm_layer=is_norm_layer))

        if isinstance(module.mlp, MoELayer):
            map_mlp_func = self._map_moe_layer
            norm_layer = module.pre_mlp_layernorm
            norm_src_key = f"{src_prefix}pre_mlp_layernorm."
            norm_dst_key = f"{dst_prefix}post_attention_layernorm."
            is_norm_layer = True
        else:
            map_mlp_func = self._map_mlp
            norm_layer = module.mlp.linear_fc1
            norm_src_key = f"{src_prefix}mlp.linear_fc1."
            norm_dst_key = f"{dst_prefix}post_attention_layernorm."
            is_norm_layer = False
        mapping.update(map_mlp_func(module.mlp, src_prefix=f"{src_prefix}mlp.", dst_prefix=f"{dst_prefix}mlp."))
        mapping.update(self._map_norm_layer(norm_layer, norm_src_key, norm_dst_key, is_norm_layer=is_norm_layer))
        return mapping

    def _map_moe_layer(self, module: 'MoELayer', src_prefix='', dst_prefix=''):
        mapping = {}
        # router
        mapping.update(self._inner_map_for_full_shape(f"{src_prefix}router.weight", f"{dst_prefix}gate.weight"))
        if module.router.enable_expert_bias:
            mapping.update(self._inner_map_for_full_shape(f"{src_prefix}router.expert_bias", f"{dst_prefix}gate.e_score_correction_bias"))

        if not module.config.moe_grouped_gemm:
            raise NotImplementedError("Parameter Sync w/ MoE SequentialMLP is not supported")
        if not isinstance(module.experts, TEGroupedMLP):
            raise NotImplementedError("Parameter Sync w/ Legacy GroupedMLP is not supported")

        # experts
        mapping.update(self._map_group_mlp(
            module.experts,
            src_prefix=f"{src_prefix}experts.",
            dst_prefix=f"{dst_prefix}experts."
        ))

        # shared experts
        if module.shared_experts is not None:
            if module.shared_experts.use_shared_expert_gate:
                mapping.update(self._inner_map_for_full_shape(f"{src_prefix}shared_experts.gate_weight", f"{dst_prefix}shared_expert_gate.weight"))
            mapping.update(self._map_mlp(
                module.shared_experts,
                src_prefix=f"{src_prefix}shared_experts.",
                dst_prefix=f"{dst_prefix}shared_experts."
            ))
        return mapping

    def _map_mlp(self, module: 'MLP', src_prefix: str='', dst_prefix: str=''):
        mapping = {}
        if not module.config.gated_linear_unit:
            raise NotImplementedError("Parameter Sync w/o GatedLinear is not supported")
        mapping.update(self._inner_map_for_gate_up_proj(
            f"{src_prefix}linear_fc1.weight",
            f"{dst_prefix}gate_up_proj.weight"
        ))
        mapping.update(self._inner_map_for_tensor_parallel(
            f"{src_prefix}linear_fc2.weight",
            f"{dst_prefix}down_proj.weight",
            mapping_type='row'
        ))
        return mapping

    def _map_group_mlp(self, module: 'TEGroupedMLP', src_prefix: str='', dst_prefix: str=''):
        # pylint: disable=unused-argument
        src_ep_rank = mpu.get_expert_model_parallel_rank()
        src_ep_size = mpu.get_expert_model_parallel_world_size()
        num_experts = self._src_arch.num_experts
        global_expert_id_start = num_experts // src_ep_size * src_ep_rank
        global_expert_id_end = num_experts // src_ep_size * (src_ep_rank + 1)
        mapping = {}
        for local_expert_id, global_expert_id in enumerate(range(global_expert_id_start, global_expert_id_end)):
            mapping.update(self._inner_map_for_gate_up_proj(
                f"{src_prefix}linear_fc1.weight{local_expert_id}",
                f"{dst_prefix}w13_weight",
                global_expert_id=global_expert_id,
                num_experts=num_experts
            ))
            mapping.update(self._inner_map_for_tensor_parallel(
                f"{src_prefix}linear_fc2.weight{local_expert_id}",
                f"{dst_prefix}w2_weight",
                global_expert_id=global_expert_id,
                num_experts=num_experts,
                mapping_type='row'
            ))
        return mapping

    def _map_mla_selfattn(self, module: 'MLASelfAttention', src_prefix: str='', dst_prefix: str=''):
        mapping = {}
        if self._src_arch.q_lora_rank is None:
            mapping.update(self._inner_map_for_tensor_parallel(
                f"{src_prefix}linear_q_proj.weight",
                f"{dst_prefix}q_proj.weight",
                mapping_type='column'
            ))
        else:
            mapping.update(self._inner_map_for_mla_down_proj(
                f"{src_prefix}linear_q_down_proj.weight",
                f"{dst_prefix}q_a_proj.weight",
            ))
            mapping.update(self._inner_map_for_tensor_parallel(
                f"{src_prefix}linear_q_up_proj.weight",
                f"{dst_prefix}q_b_proj.weight",
                mapping_type='column'
            ))
            if self._src_arch.qk_layernorm:
                mapping.update(
                    self._map_norm_layer(
                        module.linear_q_up_proj,
                        f"{src_prefix}linear_q_up_proj.",
                        f"{dst_prefix}q_a_layernorm.",
                        is_norm_layer=False
                    )
                )
        mapping.update(self._inner_map_for_mla_down_proj(
            f"{src_prefix}linear_kv_down_proj.weight",
            f"{dst_prefix}kv_a_proj_with_mqa.weight",
        ))
        mapping.update(self._inner_map_for_tensor_parallel(
            f"{src_prefix}linear_kv_up_proj.weight",
            f"{dst_prefix}kv_b_proj.weight",
            mapping_type='column'
        ))
        if self._src_arch.qk_layernorm:
            mapping.update(
                self._map_norm_layer(
                    module.linear_kv_up_proj,
                    f"{src_prefix}linear_kv_up_proj.",
                    f"{dst_prefix}kv_a_layernorm.",
                    is_norm_layer=False
                )
            )
        mapping.update(self._inner_map_for_tensor_parallel(
            f"{src_prefix}linear_proj.weight",
            f"{dst_prefix}o_proj.weight",
            mapping_type='row'
        ))
        return mapping

    def _map_selfattn(self, module: 'SelfAttention', src_prefix: str='', dst_prefix: str=''):
        mapping = {}
        if self._src_arch.qk_layernorm:
            mapping.update(self._map_norm_layer(module.q_layernorm, f"{src_prefix}q_layernorm.", f"{dst_prefix}q_norm."))
            mapping.update(self._map_norm_layer(module.k_layernorm, f"{src_prefix}k_layernorm.", f"{dst_prefix}k_norm."))

        mapping.update(self._inner_map_for_qkv_proj(
            f"{src_prefix}linear_qkv.weight",
            f"{dst_prefix}qkv_proj.weight",
        ))
        if self._src_arch.add_qkv_bias:
            mapping.update(self._inner_map_for_qkv_proj(
                f"{src_prefix}linear_qkv.bias",
                f"{dst_prefix}qkv_proj.bias",
            ))
        mapping.update(self._inner_map_for_tensor_parallel(
            f"{src_prefix}linear_proj.weight",
            f"{dst_prefix}o_proj.weight",
            mapping_type='row'
        ))
        return mapping

    def _map_preprocess_layer(self, module: 'LanguageModelEmbedding', src_prefix='', dst_prefix=''):
        if module.add_position_embedding:
            raise NotImplementedError("learned_absolute embedding is not supported")
        return self._inner_map_for_tensor_parallel(
            f"{src_prefix}word_embeddings.weight",
            f"{dst_prefix}embed_tokens.weight",
            mapping_type='column'
        )

    def _map_postprocess_layer(self, module: 'ColumnParallelLinear', src_prefix='', dst_prefix=''):
        # pylint: disable=unused-argument
        return self._inner_map_for_tensor_parallel(
            f"{src_prefix}weight",
            f"{dst_prefix}lm_head.weight",
            mapping_type='column'
        )

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
        dst_info = self._dst_name_to_metadata[dst_key]
        mapping = {}
        for src_meta, dst_meta in process_normal_tensor(
            src_info,
            self._dst_model_config.tensor_model_parallel_size,
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
        return mapping

    def _inner_map_for_full_shape(
        self,
        src_key: str,
        dst_key: str
    ):
        src_info = self._src_name_to_metadata[src_key]
        dst_info = self._dst_name_to_metadata[dst_key]
        return {
            src_info.copy(): [dst_info.copy()]
        }

    def _inner_map_for_gate_up_proj(self, src_key: str, dst_key: str, *, global_expert_id: int=None, num_experts: int=None):
        src_info = self._src_name_to_metadata[src_key]
        dst_info = self._dst_name_to_metadata[dst_key]
        mapping = {}
        for src_meta, dst_meta in process_gate_up_tensor(
            src_info,
            self._dst_model_config.tensor_model_parallel_size,
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
        return mapping

    def _inner_map_for_qkv_proj(self, src_key: str, dst_key: str):
        src_info = self._src_name_to_metadata[src_key]
        dst_info = self._dst_name_to_metadata[dst_key]
        mapping = defaultdict(list)
        for src_meta, dst_meta in process_qkv_tensor(
            src_info,
            self._src_arch.num_attention_heads,
            self._src_arch.num_query_groups,
            self._dst_model_config.tensor_model_parallel_size,
        ):
            src_meta.param_id, dst_meta.param_id = src_info.param_id, dst_info.param_id
            src_meta.dtype, dst_meta.dtype = src_info.dtype, dst_info.dtype
            mapping[src_meta].append(dst_meta)
        return mapping

    def _inner_map_for_mla_down_proj(self, src_key: str, dst_key: str):
        src_info = self._src_name_to_metadata[src_key]
        dst_info = self._dst_name_to_metadata[dst_key]
        dst_meta = src_info.refragment(1)
        dst_meta.param_id = dst_info.param_id
        dst_meta.dtype = dst_info.dtype
        return {
            src_info.copy(): [dst_meta]
        }

    @property
    def _src_arch(self):
        return self._src_model_config.megatron_model_cfg
