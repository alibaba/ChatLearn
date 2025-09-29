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
from transformers import AutoConfig

from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.experts import TEGroupedMLP

from chatlearn.configs import PolicyConfig

from .mapping_helpers import (
    VLLM_HELPERS,
    HF_HELPERS
)

from .base_megatron_mapper import BaseMegatronMapper

if TYPE_CHECKING:
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    from megatron.core.transformer.transformer_layer import TransformerLayer
    from megatron.core.tensor_parallel import ColumnParallelLinear
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention
    from megatron.core.transformer.attention import SelfAttention
    from chatlearn.models.megatron_module import MegatronModule

class MegatronLLMMapper(BaseMegatronMapper):
    """MegatronLLMMapper"""
    def __init__(
        self,
        dst_model_config: PolicyConfig,
        model: 'MegatronModule',
        *,
        mapper_config: Union[VLLM_HELPERS, HF_HELPERS] = VLLM_HELPERS,
    ):
        """The Mapper for Megatron sync. In each remote Megatron Actor,
        the method of this class is called to generate the parameter mapping
        between src and dst. Currently, the mapper supports mapping
        MCore Model to vLLM or HF Model.

        WARNING: The mapper assumes that the weights name of same
        submodules in different vLLM models are still same.

        Args:
            dst_model_config (PolicyConfig): The config of target model to
                be sychronized
            model (MegatronModule): The source Megatron Module
            mapper_config (Union[VLLM_HELPERS, HF_HELPERS]): The mapping mode.
        """
        super().__init__(dst_model_config=dst_model_config, model=model, mapper_config=mapper_config)

    def _map_llm_model(self, model: nn.Module, vp_stage: int, layer_offset: int):
        if model.pre_process:
            self._map_preprocess_layer(
                model.embedding,
                src_prefix=f"{vp_stage}-embedding.",
                dst_prefix="model.",
            )

        for layer_idx in range(model.decoder.num_layers_per_pipeline_rank):
            global_layer_id = layer_offset + layer_idx
            self._map_decoder_layer(
                model.decoder.layers[layer_idx],
                src_prefix=f"{vp_stage}-decoder.layers.{layer_idx}.",
                dst_prefix=f"model.layers.{global_layer_id}.",
            )

        if model.post_process:
            self._map_norm_layer(
                model.decoder.final_layernorm,
                src_prefix=f"{vp_stage}-decoder.final_layernorm.",
                dst_prefix="model.norm.",
            )

            if model.share_embeddings_and_output_weights and model.pre_process:
                self._map_postprocess_layer(
                    model.embedding,
                    src_prefix=f"{vp_stage}-embedding.word_embeddings.",
                    dst_prefix="",
                )
            else:
                self._map_postprocess_layer(
                    model.output_layer,
                    src_prefix=f"{vp_stage}-output_layer.",
                    dst_prefix="",
                )
    
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

            self._map_llm_model(model, vp_stage=vp_stage, layer_offset=layer_offset)

        mapping = self._mapping
        self._mapping = None
        return mapping

    def _map_norm_layer(self, module: nn.Module, src_prefix: str='', dst_prefix: str='', *, is_norm_layer: bool=True):
        """If is_norm_layer is True, try to map on all possible keys,
        otherwise only map on `layer_norm_weight` and `layer_norm_bias`
        """
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
            self._inner_map_for_full_shape(
                f"{src_prefix}{item}",
                f"{dst_prefix}{_keynames[item]}"
            )

    def _map_decoder_layer(self, module: 'TransformerLayer', src_prefix: str='', dst_prefix: str=''):
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
        map_attn_func(module.self_attention, src_prefix=f"{src_prefix}self_attention.", dst_prefix=f"{dst_prefix}self_attn.")
        self._map_norm_layer(norm_layer, norm_src_key, norm_dst_key, is_norm_layer=is_norm_layer)

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
        map_mlp_func(module.mlp, src_prefix=f"{src_prefix}mlp.", dst_prefix=f"{dst_prefix}mlp.")
        self._map_norm_layer(norm_layer, norm_src_key, norm_dst_key, is_norm_layer=is_norm_layer)

    def _map_moe_layer(self, module: 'MoELayer', src_prefix='', dst_prefix=''):
        mapping = {}
        # router
        self._inner_map_for_full_shape(f"{src_prefix}router.weight", f"{dst_prefix}gate.weight")
        if module.router.enable_expert_bias:
            self._inner_map_for_full_shape(f"{src_prefix}router.expert_bias", f"{dst_prefix}gate.e_score_correction_bias")

        if not module.config.moe_grouped_gemm:
            raise NotImplementedError("Parameter Sync w/ MoE SequentialMLP is not supported")
        if not isinstance(module.experts, TEGroupedMLP):
            raise NotImplementedError("Parameter Sync w/ Legacy GroupedMLP is not supported")

        # experts
        self._map_group_mlp(
            module.experts,
            src_prefix=f"{src_prefix}experts.",
            dst_prefix=f"{dst_prefix}experts."
        )

        # shared experts
        if module.shared_experts is not None:
            if module.shared_experts.use_shared_expert_gate:
                self._inner_map_for_full_shape(
                    f"{src_prefix}shared_experts.gate_weight",
                    f"{dst_prefix}shared_expert_gate.weight"
                )
            # NOTE: if transformer.config have n_shared_experts, mapping to `shared_experts`, otherwise `shared_expert`
            # `shared_experts`: DeepSeek-V2, DeepSeek-V3, etc.
            # `shared_expert`: Qwen2-MoE, LLaMA-4, etc.
            hf_config = AutoConfig.from_pretrained(self._dst_model_config.load, trust_remote_code=self._dst_model_config.trust_remote_code)
            shared_expert_key = 'shared_experts' if hasattr(hf_config, 'n_shared_experts') else 'shared_expert'
            self._map_mlp(
                module.shared_experts,
                src_prefix=f"{src_prefix}shared_experts.",
                dst_prefix=f"{dst_prefix}{shared_expert_key}."
            )
        return mapping

    def _map_mlp(self, module: 'MLP', src_prefix: str='', dst_prefix: str='', is_vision_block=False):
        if not module.config.gated_linear_unit:
            raise NotImplementedError("Parameter Sync w/o GatedLinear is not supported")

        dst_names = ['gate_proj', 'up_proj']
        if self._mapper_config.merge_gate_up and not is_vision_block:
            dst_names = ['gate_up_proj']

        for dst_name in dst_names:
            self._inner_map_for_gate_up_proj(
                f"{src_prefix}linear_fc1.weight",
                f"{dst_prefix}{dst_name}.weight",
                proj_type=dst_name
            )

            if module.config.add_bias_linear:
                self._inner_map_for_gate_up_proj(
                    f"{src_prefix}linear_fc1.bias",
                    f"{dst_prefix}{dst_name}.bias",
                    proj_type=dst_name
                )

        self._inner_map_for_tensor_parallel(
            f"{src_prefix}linear_fc2.weight",
            f"{dst_prefix}down_proj.weight",
            mapping_type='row'
        )

        if module.config.add_bias_linear:
            self._inner_map_for_full_shape(
                f"{src_prefix}linear_fc2.bias",
                f"{dst_prefix}down_proj.bias"
            )

    def _map_group_mlp(self, module: 'TEGroupedMLP', src_prefix: str='', dst_prefix: str=''):
        # pylint: disable=unused-argument
        src_ep_rank = mpu.get_expert_model_parallel_rank()
        src_ep_size = mpu.get_expert_model_parallel_world_size()
        num_experts = self._src_arch.num_experts
        global_expert_id_start = num_experts // src_ep_size * src_ep_rank
        global_expert_id_end = num_experts // src_ep_size * (src_ep_rank + 1)
        for local_expert_id, global_expert_id in enumerate(range(global_expert_id_start, global_expert_id_end)):
            if self._mapper_config.merge_expert:
                if not self._mapper_config.merge_gate_up:
                    raise NotImplementedError("merge_expert w/o merge_gate_up is not implemented.")
                self._inner_map_for_gate_up_proj(
                    f"{src_prefix}linear_fc1.weight{local_expert_id}",
                    f"{dst_prefix}w13_weight",
                    proj_type='gate_up_proj',
                    global_expert_id=global_expert_id,
                    num_experts=num_experts
                )
                self._inner_map_for_tensor_parallel(
                    f"{src_prefix}linear_fc2.weight{local_expert_id}",
                    f"{dst_prefix}w2_weight",
                    global_expert_id=global_expert_id,
                    num_experts=num_experts,
                    mapping_type='row'
                )
            else:
                if self._mapper_config.merge_gate_up:
                    raise NotImplementedError("no merge_expert w/ merge_gate_up is not implemented.")
                for dst_name in ['gate_proj', 'up_proj']:
                    self._inner_map_for_gate_up_proj(
                        f"{src_prefix}linear_fc1.weight{local_expert_id}",
                        f"{dst_prefix}{global_expert_id}.{dst_name}.weight",
                        proj_type=dst_name,
                    )
                self._inner_map_for_tensor_parallel(
                    f"{src_prefix}linear_fc2.weight{local_expert_id}",
                    f"{dst_prefix}{global_expert_id}.down_proj.weight",
                    mapping_type='row'
                )

    def _map_mla_selfattn(self, module: 'MLASelfAttention', src_prefix: str='', dst_prefix: str=''):
        if self._src_arch.q_lora_rank is None:
            self._inner_map_for_tensor_parallel(
                f"{src_prefix}linear_q_proj.weight",
                f"{dst_prefix}q_proj.weight",
                mapping_type='column'
            )
        else:
            self._inner_map_for_mla_down_proj(
                f"{src_prefix}linear_q_down_proj.weight",
                f"{dst_prefix}q_a_proj.weight",
            )
            self._inner_map_for_tensor_parallel(
                f"{src_prefix}linear_q_up_proj.weight",
                f"{dst_prefix}q_b_proj.weight",
                mapping_type='column'
            )
            if self._src_arch.qk_layernorm:
                self._map_norm_layer(
                    module.linear_q_up_proj,
                    f"{src_prefix}linear_q_up_proj.",
                    f"{dst_prefix}q_a_layernorm.",
                    is_norm_layer=False
                )
        self._inner_map_for_mla_down_proj(
            f"{src_prefix}linear_kv_down_proj.weight",
            f"{dst_prefix}kv_a_proj_with_mqa.weight",
        )
        self._inner_map_for_tensor_parallel(
            f"{src_prefix}linear_kv_up_proj.weight",
            f"{dst_prefix}kv_b_proj.weight",
            mapping_type='column'
        )
        if self._src_arch.qk_layernorm:
            self._map_norm_layer(
                module.linear_kv_up_proj,
                f"{src_prefix}linear_kv_up_proj.",
                f"{dst_prefix}kv_a_layernorm.",
                is_norm_layer=False
            )
        self._inner_map_for_tensor_parallel(
            f"{src_prefix}linear_proj.weight",
            f"{dst_prefix}o_proj.weight",
            mapping_type='row'
        )

    def _map_selfattn(self, module: 'SelfAttention', src_prefix: str='', dst_prefix: str=''):
        if self._src_arch.qk_layernorm:
            self._map_norm_layer(module.q_layernorm, f"{src_prefix}q_layernorm.", f"{dst_prefix}q_norm.")
            self._map_norm_layer(module.k_layernorm, f"{src_prefix}k_layernorm.", f"{dst_prefix}k_norm.")

        dst_names = ['q_proj', 'k_proj', 'v_proj']
        if self._mapper_config.merge_qkv:
            dst_names = ['qkv_proj']

        for dst_name in dst_names:
            self._inner_map_for_qkv_proj(
                f"{src_prefix}linear_qkv.weight",
                f"{dst_prefix}{dst_name}.weight",
                proj_type=dst_name,
                num_attention_heads = self._src_arch.num_attention_heads,
                num_query_groups = self._src_arch.num_query_groups
            )
            if self._src_arch.add_qkv_bias:
                self._inner_map_for_qkv_proj(
                    f"{src_prefix}linear_qkv.bias",
                    f"{dst_prefix}{dst_name}.bias",
                    proj_type=dst_name,
                    num_attention_heads = self._src_arch.num_attention_heads,
                    num_query_groups = self._src_arch.num_query_groups
                )

        self._inner_map_for_tensor_parallel(
            f"{src_prefix}linear_proj.weight",
            f"{dst_prefix}o_proj.weight",
            mapping_type='row'
        )

    def _map_preprocess_layer(self, module: 'LanguageModelEmbedding', src_prefix='', dst_prefix=''):
        if module.add_position_embedding:
            raise NotImplementedError("learned_absolute embedding is not supported")
        self._inner_map_for_tensor_parallel(
            f"{src_prefix}word_embeddings.weight",
            f"{dst_prefix}embed_tokens.weight",
            mapping_type='column'
        )

    def _map_postprocess_layer(self, module: 'ColumnParallelLinear', src_prefix='', dst_prefix=''):
        # pylint: disable=unused-argument
        if (
            not self._src_arch.untie_embeddings_and_output_weights and
            f"{dst_prefix}lm_head.weight" not in self._dst_name_to_metadata
        ):
            return
        self._inner_map_for_tensor_parallel(
            f"{src_prefix}weight",
            f"{dst_prefix}lm_head.weight",
            mapping_type='column'
        )
