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
from typing import TYPE_CHECKING, Union, Dict

import inspect
from torch import nn
from transformers import AutoConfig

from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.ssm.mamba_block import MambaStack
from megatron.core.ssm.mamba_layer import MambaLayer

from chatlearn.configs import PolicyConfig

from .mapping_helpers import (
    VLLM_HELPERS,
    HF_HELPERS
)
from .metadata import (
    SelfAttnKeyMapping,
    MLPKeyMapping,
    DecoderLayerKeyMapping,
    LanguageModelKeyMapping,
    MoELayerKeyMapping,
    MLASelfAttnKeyMapping
)
from .base_megatron_mapper import BaseMegatronMapper

from chatlearn.utils.logger import logger

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
        """The Mapper for Megatron LLM sync.

        Args:
            dst_model_config (PolicyConfig): The config of target model to
                be sychronized
            model (MegatronModule): The source Megatron Module
            mapper_config (Union[VLLM_HELPERS, HF_HELPERS]): The mapping mode.
        """
        super().__init__(dst_model_config=dst_model_config, model=model, mapper_config=mapper_config)

    # NOTE: the following function implements the module-wise sync mapping
    def _build_layer_index_mapping(self, decoder, vp_stage):
        """
            Map the local layer index (ranged from 0 to model.decoder.num_layers_per_pipeline_rank)
            to the global huggingface layer index
        """
        if isinstance(decoder, TransformerBlock):
            layer_offset = get_transformer_layer_offset(decoder.config, vp_stage=vp_stage)
            num_layers_per_pipeline_rank = decoder.num_layers_per_pipeline_rank
            return {n: layer_offset + n for n in range(num_layers_per_pipeline_rank)}
        elif isinstance(decoder, MambaStack):
            assert vp_stage == 0, "Mamba do not support VPP"
            # NOTE: currently we assume MambaLayer just replaces some of Attention
            # layout should be: ((mamba | attn) mlp) x n
            num_layers_per_pipeline_rank = decoder.num_layers_per_pipeline_rank
            layer_offset = num_layers_per_pipeline_rank * mpu.get_pipeline_model_parallel_rank()
            return {n: (n + layer_offset) // 2 for n in range(num_layers_per_pipeline_rank)}
        else:
            raise ValueError(f"Unexpected decoder type: {type(decoder)}")

    def _map_model(self):
        """Mapping the local name of src model to global name of
        dst model
        """
        cfg = LanguageModelKeyMapping(
            decoder_layer_cfg=DecoderLayerKeyMapping(
                self_attn_cfg=SelfAttnKeyMapping(use_merged_qkv=self._mapper_config.merge_qkv),
                mlp_cfg=MLPKeyMapping(use_merged_gate_up=self._mapper_config.merge_gate_up)
            )
        )
        for vp_stage, model in enumerate(self.model):
            if getattr(model, 'mtp_process', False):
                raise NotImplementedError("Currently, the mapper does not support MTP")

            self._map_llm_model(
                model,
                cfg,
                index_mapping=self._build_layer_index_mapping(
                    model.decoder,
                    vp_stage
                ),
                src_prefix=f"{vp_stage}-",
                dst_prefix=""
            )

        mapping = self._mapping
        self._mapping = None
        return mapping

    def _map_llm_model(
        self,
        model: nn.Module,
        cfg: LanguageModelKeyMapping,
        index_mapping: Dict[int, int],
        src_prefix: str='',
        dst_prefix: str=''
    ):
        if model.pre_process:
            self._map_preprocess_layer(
                model.embedding,
                src_prefix=f"{src_prefix}embedding.",
                dst_prefix=f"{dst_prefix}{cfg.word_embeddings}",
            )

        for layer_idx in range(model.decoder.num_layers_per_pipeline_rank):
            global_layer_id = index_mapping[layer_idx]
            if isinstance(model.decoder.layers[layer_idx], MambaLayer):
                self._map_mamba_layer(
                    model.decoder.layers[layer_idx],
                    src_prefix=f"{src_prefix}decoder.layers.{layer_idx}.",
                    dst_prefix=f"{dst_prefix}{cfg.decoder_layer}{global_layer_id}.",
                )
            else:
                self._map_transformer_layer(
                    model.decoder.layers[layer_idx],
                    cfg=cfg.decoder_layer_cfg,
                    src_prefix=f"{src_prefix}decoder.layers.{layer_idx}.",
                    dst_prefix=f"{dst_prefix}{cfg.decoder_layer}{global_layer_id}.",
                )

        if model.post_process:
            if isinstance(model.decoder, MambaStack):
                self._map_norm_layer(
                    model.decoder.final_norm,
                    src_prefix=f"{src_prefix}decoder.final_norm.",
                    dst_prefix=f"{dst_prefix}{cfg.final_layernorm}",
                )
            else:
                self._map_norm_layer(
                    model.decoder.final_layernorm,
                    src_prefix=f"{src_prefix}decoder.final_layernorm.",
                    dst_prefix=f"{dst_prefix}{cfg.final_layernorm}",
                )

            if model.share_embeddings_and_output_weights and model.pre_process:
                self._map_postprocess_layer(
                    model.embedding,
                    src_prefix=f"{src_prefix}embedding.word_embeddings.",
                    dst_prefix=f"{dst_prefix}{cfg.output_layer}",
                )
            else:
                self._map_postprocess_layer(
                    model.output_layer,
                    src_prefix=f"{src_prefix}output_layer.",
                    dst_prefix=f"{dst_prefix}{cfg.output_layer}",
                )

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

    def _map_transformer_layer(self, module: 'TransformerLayer', cfg: DecoderLayerKeyMapping, src_prefix: str='', dst_prefix: str=''):
        submodule_config = module.submodules_config
        has_self_attention = submodule_config.self_attention is not IdentityOp
        has_mlp = submodule_config.mlp is not IdentityOp
        assert has_self_attention or has_mlp, f"The TransformerLayer should at least contains one of self_attn or mlp!"

        if has_self_attention:
            if module.config.multi_latent_attention:
                map_attn_func = self._map_mla_selfattn
                norm_layer = module.input_layernorm
                norm_src_key = f"{src_prefix}input_layernorm."
                is_norm_layer = True
            else:
                map_attn_func = self._map_selfattn
                is_gated_attention = hasattr(module.self_attention, 'linear_qgkv')
                if is_gated_attention:
                    norm_layer = module.self_attention.linear_qgkv
                    norm_src_key = f"{src_prefix}self_attention.linear_qgkv."
                else:
                    norm_layer = module.self_attention.linear_qkv
                    norm_src_key = f"{src_prefix}self_attention.linear_qkv."
                is_norm_layer = False
            map_attn_func(
                module.self_attention,
                cfg=cfg.self_attn_cfg,
                src_prefix=f"{src_prefix}self_attention.",
                dst_prefix=f"{dst_prefix}{cfg.self_attn}",
            )
            self._map_norm_layer(
                norm_layer,
                norm_src_key,
                dst_prefix=f"{dst_prefix}{cfg.input_layernorm}",
                is_norm_layer=is_norm_layer
            )

        if has_mlp:
            if isinstance(module.mlp, MoELayer):
                map_mlp_func = self._map_moe_layer
                norm_layer = module.pre_mlp_layernorm
                norm_src_key = f"{src_prefix}pre_mlp_layernorm."
                is_norm_layer = True
            else:
                map_mlp_func = self._map_mlp
                norm_layer = module.mlp.linear_fc1
                norm_src_key = f"{src_prefix}mlp.linear_fc1."
                is_norm_layer = False
            map_mlp_func(
                module.mlp,
                cfg=cfg.mlp_cfg,
                src_prefix=f"{src_prefix}mlp.",
                dst_prefix=f"{dst_prefix}{cfg.mlp}",
            )
            self._map_norm_layer(
                norm_layer,
                norm_src_key,
                dst_prefix=f"{dst_prefix}{cfg.pre_mlp_layernorm}",
                is_norm_layer=is_norm_layer
            )

    def _map_mamba_layer(self, module, src_prefix='', dst_prefix=''):
        # NOTE: the API is experimental as MambaLayer is not general enough currently
        self._map_norm_layer(
            module.mixer.in_proj,
            f"{src_prefix}mixer.in_proj.",
            dst_prefix=f"{dst_prefix}input_layernorm.",
            is_norm_layer=False
        )
        self._map_mamba_mixer(
            module.mixer,
            src_prefix=f"{src_prefix}mixer.",
            dst_prefix=f"{dst_prefix}linear_attn.",
        )

    def _map_mamba_mixer(self, module,  src_prefix='', dst_prefix=''):
        Nk, Nv, Dk, Dv = (
            module.ngroups,
            module.nheads,
            module.d_state,
            module.headdim
        )

        # in_proj
        src_layout = [
            ('z', Dv * Nv), 
            ('v', Dv * Nv), 
            ('q', Dk * Nk), 
            ('k', Dk * Nk), 
            ('b', Nv), 
            ('a', Nv)
        ]
        self._inner_map_for_linear_attn(
            f"{src_prefix}in_proj.weight",
            f"{dst_prefix}in_proj_qkvz.weight",
            src_layout=src_layout,
            required_layout=['q', 'k', 'v', 'z'],
            n_groups=Nk
        )
        self._inner_map_for_linear_attn(
            f"{src_prefix}in_proj.weight",
            f"{dst_prefix}in_proj_ba.weight",
            src_layout=src_layout,
            required_layout=['b', 'a'],
            n_groups=Nk
        )
        # conv1d
        src_layout = [
            ('conv_v', Dv * Nv), 
            ('conv_q', Dk * Nk), 
            ('conv_k', Dk * Nk), 
        ]
        self._inner_map_for_merged_linear(
            f"{src_prefix}conv1d.weight",
            f"{dst_prefix}conv1d.weight",
            src_layout=src_layout,
            required_layout=['conv_q', 'conv_k', 'conv_v']
        )
        self._inner_map_for_tensor_parallel(
            f"{src_prefix}dt_bias",
            f"{dst_prefix}dt_bias",
            mapping_type='column'
        )

        self._inner_map_for_tensor_parallel(
            f"{src_prefix}A_log",
            f"{dst_prefix}A_log",
            mapping_type='column'
        )
        if module.D is not None:
            raise NotImplementedError()

        self._map_norm_layer(
            module.norm,
            f"{src_prefix}norm.",
            dst_prefix=f"{dst_prefix}norm.",
            is_norm_layer=True
        )
        self._inner_map_for_tensor_parallel(
            f"{src_prefix}out_proj.weight",
            f"{dst_prefix}out_proj.weight",
            mapping_type='row'
        )

    def _map_moe_layer(self, module: 'MoELayer', cfg: MoELayerKeyMapping, src_prefix='', dst_prefix=''):
        # pylint: disable=unused-argument
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
                cfg=MLPKeyMapping(use_merged_gate_up=self._mapper_config.merge_gate_up),
                src_prefix=f"{src_prefix}shared_experts.",
                dst_prefix=f"{dst_prefix}{shared_expert_key}."
            )
        return mapping

    def _map_mlp(
        self,
        module: 'MLP',
        cfg: MLPKeyMapping,
        src_prefix: str='',
        dst_prefix: str='',
    ):
        param_types = ['weight']
        if module.config.add_bias_linear:
            param_types = ['weight', 'bias']

        for param_type in param_types:
            if not module.config.gated_linear_unit:
                self._inner_map_for_tensor_parallel(
                    f"{src_prefix}linear_fc1.{param_type}",
                    f"{dst_prefix}{cfg.up_proj}{param_type}",
                    mapping_type='column'
                )
            else:
                dst_names = {'gate_proj': cfg.gate_proj, 'up_proj': cfg.up_proj}
                if cfg.use_merged_gate_up:
                    dst_names = {'gate_up_proj': cfg.gate_up_proj}

                for dst_type, dst_name in dst_names.items():
                    self._inner_map_for_gate_up_proj(
                        f"{src_prefix}linear_fc1.{param_type}",
                        f"{dst_prefix}{dst_name}{param_type}",
                        proj_type=dst_type
                    )
            self._inner_map_for_tensor_parallel(
                f"{src_prefix}linear_fc2.{param_type}",
                f"{dst_prefix}{cfg.down_proj}{param_type}",
                mapping_type='row'
            )

    def _map_group_mlp(self, module: 'TEGroupedMLP', src_prefix: str='', dst_prefix: str=''):
        # pylint: disable=unused-argument
        src_ep_rank = mpu.get_expert_model_parallel_rank()
        src_ep_size = mpu.get_expert_model_parallel_world_size()
        num_experts = module.config.num_moe_experts
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

    def _map_mla_selfattn(self, module: 'MLASelfAttention', cfg: MLASelfAttnKeyMapping, src_prefix: str='', dst_prefix: str=''):
        # pylint: disable=unused-argument
        if module.config.q_lora_rank is None:
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
            if module.config.qk_layernorm:
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
        if module.config.qk_layernorm:
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

    def _map_selfattn(
        self,
        module: 'SelfAttention',
        cfg: SelfAttnKeyMapping,
        src_prefix: str='',
        dst_prefix: str=''
    ):
        if module.config.qk_layernorm:
            self._map_norm_layer(module.q_layernorm, f"{src_prefix}q_layernorm.", f"{dst_prefix}{cfg.q_layernorm}")
            self._map_norm_layer(module.k_layernorm, f"{src_prefix}k_layernorm.", f"{dst_prefix}{cfg.k_layernorm}")

        qkv_dst_names = {'qkv_proj': cfg.qkv_proj}
        if not cfg.use_merged_qkv:
            qkv_dst_names = {'q_proj': cfg.q_proj, 'k_proj': cfg.k_proj, 'v_proj': cfg.v_proj}

        param_types = ['weight']
        if module.config.add_qkv_bias:
            param_types = ['weight', 'bias']

        # TODO: make better condition
        is_gated_attention = hasattr(module, 'linear_qgkv')
        for param_type in param_types:
            for dst_type, dst_name in qkv_dst_names.items():
                if is_gated_attention:
                    src_key = f"{src_prefix}linear_qgkv.{param_type}"
                else:
                    src_key = f"{src_prefix}linear_qkv.{param_type}"
                self._inner_map_for_qkv_proj(
                    src_key,
                    f"{dst_prefix}{dst_name}{param_type}",
                    proj_type=dst_type,
                    num_attention_heads = module.config.num_attention_heads,
                    num_query_groups = module.config.num_query_groups,
                    is_gated_attention = is_gated_attention
                )

        param_types = ['weight']
        if module.config.add_bias_linear:
            param_types = ['weight', 'bias']
        for param_type in param_types:
            self._inner_map_for_tensor_parallel(
                f"{src_prefix}linear_proj.{param_type}",
                f"{dst_prefix}{cfg.out_proj}{param_type}",
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
            not self._src_model_config.megatron_model_cfg.untie_embeddings_and_output_weights and
            f"{dst_prefix}lm_head.weight" not in self._dst_name_to_metadata
        ):
            return
        self._inner_map_for_tensor_parallel(
            f"{src_prefix}weight",
            f"{dst_prefix}lm_head.weight",
            mapping_type='column'
        )
