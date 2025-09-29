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
from typing import TYPE_CHECKING, Union

import inspect
from torch import nn

from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from chatlearn.configs import PolicyConfig

from .mapping_helpers import (
    VLLM_HELPERS,
    HF_HELPERS
)

from .megatron_llm_mapper import MegatronLLMMapper

if TYPE_CHECKING:
    from chatlearn.models.megatron_module import MegatronModule
    from megatron.core.transformer.transformer_layer import TransformerLayer

class MegatronVLMMapper(MegatronLLMMapper):
    """MegatronVLMMapper"""
    def __init__(
        self,
        dst_model_config: PolicyConfig,
        model: 'MegatronModule',
        *,
        mapper_config: Union[VLLM_HELPERS, HF_HELPERS] = VLLM_HELPERS,
    ):
        """The Mapper for Megatron VLM sync.

        Args:
            dst_model_config (PolicyConfig): The config of target model to
                be sychronized
            model (MegatronModule): The source Megatron Module
            mapper_config (Union[VLLM_HELPERS, HF_HELPERS]): The mapping mode.
        """
        super().__init__(dst_model_config=dst_model_config, model=model, mapper_config=mapper_config)

    # NOTE: the following function implements the module-wise sync mapping
    def _map_model(self):
        """Mapping the local name of src model to global name of
        dst model
        """
        for vp_stage, model in enumerate(self.model):
            if 'vp_stage' in inspect.signature(get_transformer_layer_offset).parameters:
                layer_offset = get_transformer_layer_offset(model.config, vp_stage=vp_stage)
            else:
                if len(self.model) > 1:
                    mpu.set_virtual_pipeline_model_parallel_rank(vp_stage)
                layer_offset = get_transformer_layer_offset(model.config)
                if len(self.model) > 1:
                    mpu.set_virtual_pipeline_model_parallel_rank(None)

            # TODO: VLM model does not have mtp_process, fix it in Pai-Megatron-Patch
            if getattr(model, 'mtp_process', None):
                raise NotImplementedError("Currently, the mapper does not support MTP")

            if hasattr(model, 'vision_model'):
                self._map_vlm_model(model, vp_stage=vp_stage, layer_offset=layer_offset)
            else:
                # llm model
                self._map_llm_model(model, vp_stage=vp_stage, layer_offset=layer_offset)

        mapping = self._mapping
        self._mapping = None
        return mapping

    def _map_vlm_model(self, model: nn.Module, vp_stage: int, layer_offset: int):
        dst_language_prefix = self._mapper_config.dst_language_prefix
        dst_vision_prefix = self._mapper_config.dst_vision_prefix
        dst_lm_head_prefix = self._mapper_config.dst_lm_head_prefix

        if model.pre_process:
            self._map_preprocess_layer(
                model.language_model.embedding,
                src_prefix=f"{vp_stage}-language_model.embedding.",
                dst_prefix=f"{dst_language_prefix}",
            )

            self._inner_map_for_full_shape(
                f"{vp_stage}-vision_model.patch_embed.proj.weight",
                f"{dst_vision_prefix}patch_embed.proj.weight"
            )

            # vision model decoder
            for layer_idx in range(model.vision_config.num_layers):
                global_layer_id = layer_offset + layer_idx
                self._map_vision_layer(
                    model.vision_model.decoder.layers[layer_idx],
                    src_prefix=f"{vp_stage}-vision_model.decoder.layers.{layer_idx}.",
                    dst_prefix=f"{dst_vision_prefix}blocks.{global_layer_id}.",
                    num_attention_heads=model.vision_config.num_attention_heads,
                    num_query_groups=model.vision_config.num_query_groups
                )

            # vision model projection
            self._inner_map_for_full_shape(
                f"{vp_stage}-vision_model.decoder.final_layernorm.weight",
                f"{dst_vision_prefix}merger.ln_q.weight"
            )

            self._inner_map_for_tensor_parallel(
                f"{vp_stage}-vision_model.projection.encoder.linear_fc1.weight",
                f"{dst_vision_prefix}merger.mlp.0.weight",
                mapping_type='column'
            )

            self._inner_map_for_tensor_parallel(
                f"{vp_stage}-vision_model.projection.encoder.linear_fc1.bias",
                f"{dst_vision_prefix}merger.mlp.0.bias",
                mapping_type='column'
            )

            self._inner_map_for_tensor_parallel(
                f"{vp_stage}-vision_model.projection.encoder.linear_fc2.weight",
                f"{dst_vision_prefix}merger.mlp.2.weight",
                mapping_type='row'
            )

            # bias for row is not slice, so we need to map it to full shape
            self._inner_map_for_full_shape(
                f"{vp_stage}-vision_model.projection.encoder.linear_fc2.bias",
                f"{dst_vision_prefix}merger.mlp.2.bias"
            )

        for layer_idx in range(model.language_model.decoder.num_layers_per_pipeline_rank):
            global_layer_id = layer_offset + layer_idx
            self._map_decoder_layer(
                model.language_model.decoder.layers[layer_idx],
                src_prefix=f"{vp_stage}-language_model.decoder.layers.{layer_idx}.",
                dst_prefix=f"{dst_language_prefix}layers.{global_layer_id}.",
            )

        if model.post_process:
            self._map_norm_layer(
                model.language_model.decoder.final_layernorm,
                src_prefix=f"{vp_stage}-language_model.decoder.final_layernorm.",
                dst_prefix=f"{dst_language_prefix}norm.",
            )

            if model.share_embeddings_and_output_weights and model.pre_process:
                self._map_postprocess_layer(
                    model.language_model.embedding,
                    src_prefix=f"{vp_stage}-language_model.embedding.word_embeddings.",
                    dst_prefix=f"{dst_lm_head_prefix}",
                )
            else:
                self._map_postprocess_layer(
                    model.language_model.output_layer,
                    src_prefix=f"{vp_stage}-language_model.output_layer.",
                    dst_prefix=f"{dst_lm_head_prefix}",
                )

    def _map_vision_layer(
        self,
        module: 'TransformerLayer',
        src_prefix: str = '',
        dst_prefix: str = '',
        num_attention_heads: int = None,
        num_query_groups: int = None
    ):
        # module.self_attention
        # linear_proj
        self._inner_map_for_tensor_parallel(
            f"{src_prefix}self_attention.linear_proj.weight",
            f"{dst_prefix}attn.proj.weight",
            mapping_type='row'
        )

        # bias for row is not slice, so we need to map it to full shape
        self._inner_map_for_full_shape(
            f"{src_prefix}self_attention.linear_proj.bias",
            f"{dst_prefix}attn.proj.bias"
        )

        # linear_qkv
        self._inner_map_for_qkv_proj(
            f"{src_prefix}self_attention.linear_qkv.weight",
            f"{dst_prefix}attn.qkv.weight",
            proj_type='qkv_proj',
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups
        )
        if module.config.add_qkv_bias:
            self._inner_map_for_qkv_proj(
                f"{src_prefix}self_attention.linear_qkv.bias",
                f"{dst_prefix}attn.qkv.bias",
                proj_type='qkv_proj',
                num_attention_heads=num_attention_heads,
                num_query_groups=num_query_groups
            )

        # linear_qkv_norm
        self._inner_map_for_full_shape(
            f"{src_prefix}self_attention.linear_qkv.layer_norm_weight",
            f"{dst_prefix}norm1.weight"
        )

        # module.mlp
        self._map_mlp(module.mlp, src_prefix=f"{src_prefix}mlp.", dst_prefix=f"{dst_prefix}mlp.", is_vision_block=True)

        # mlp norm
        self._inner_map_for_full_shape(
            f"{src_prefix}mlp.linear_fc1.layer_norm_weight",
            f"{dst_prefix}norm2.weight"
        )
