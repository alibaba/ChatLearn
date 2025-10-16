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
from .metadata import (
    SelfAttnKeyMapping,
    MLPKeyMapping,
    DecoderLayerKeyMapping,
    LanguageModelKeyMapping
)

from .megatron_llm_mapper import MegatronLLMMapper

if TYPE_CHECKING:
    from chatlearn.models.megatron_module import MegatronModule

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
        # TODO: clean this config object
        cfg = LanguageModelKeyMapping(
            word_embeddings=self._mapper_config.dst_language_prefix,
            decoder_layer=f"{self._mapper_config.dst_language_prefix}layers.",
            decoder_layer_cfg=DecoderLayerKeyMapping(
                self_attn_cfg=SelfAttnKeyMapping(use_merged_qkv=self._mapper_config.merge_qkv),
                mlp_cfg=MLPKeyMapping(use_merged_gate_up=self._mapper_config.merge_gate_up)
            ),
            final_layernorm=f"{self._mapper_config.dst_language_prefix}norm.",
            output_layer=self._mapper_config.dst_lm_head_prefix
        )

        for vp_stage, model in enumerate(self.model):
            if getattr(model, 'mtp_process', False):
                raise NotImplementedError("Currently, the mapper does not support MTP")

            if hasattr(model, 'vision_model'):
                # assert layer_offset == 0
                self._map_vision_model(
                    model.vision_model,
                    src_prefix=f"{vp_stage}-vision_model.",
                    dst_prefix=self._mapper_config.dst_vision_prefix
                )

            # llm model
            self._map_llm_model(
                model.language_model,
                cfg=cfg,
                index_mapping=self._build_layer_index_mapping(
                    model.language_model.decoder,
                    vp_stage
                ),
                src_prefix=f"{vp_stage}-language_model.",
                dst_prefix=""
            )

        mapping = self._mapping
        self._mapping = None
        return mapping

    def _map_vision_model(self,
        model: nn.Module,
        src_prefix: str = '',
        dst_prefix: str = ''
    ):
        self._inner_map_for_full_shape(
            f"{src_prefix}patch_embed.proj.weight",
            f"{dst_prefix}patch_embed.proj.weight"
        )

        # vision model decoder
        decoder_layer_cfg = DecoderLayerKeyMapping(
            input_layernorm='norm1.',
            self_attn='attn.',
            self_attn_cfg=SelfAttnKeyMapping(
                qkv_proj='qkv.',
                out_proj='proj.',
                use_merged_qkv=True
            ),
            pre_mlp_layernorm='norm2.'
        )
        for layer_idx in range(model.config.num_layers):
            self._map_transformer_layer(
                model.decoder.layers[layer_idx],
                decoder_layer_cfg,
                src_prefix=f"{src_prefix}decoder.layers.{layer_idx}.",
                dst_prefix=f"{dst_prefix}blocks.{layer_idx}.",
            )

        # vision model projection
        self._inner_map_for_full_shape(
            f"{src_prefix}decoder.final_layernorm.weight",
            f"{dst_prefix}merger.ln_q.weight"
        )
        mlp_cfg = MLPKeyMapping(up_proj='0.', down_proj='2.')
        self._map_mlp(
            model.projection.encoder,
            mlp_cfg,
            src_prefix=f"{src_prefix}projection.encoder.",
            dst_prefix=f"{dst_prefix}merger.mlp."
        )
