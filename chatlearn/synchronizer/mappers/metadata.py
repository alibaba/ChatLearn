# pylint: disable=missing-module-docstring, missing-class-docstring
from typing import Union
from dataclasses import dataclass, field


@dataclass
class SelfAttnKeyMapping:
    q_layernorm: str = 'q_norm.'
    k_layernorm: str = 'k_norm.'
    qkv_proj: str = 'qkv_proj.'
    q_proj: str = 'q_proj.'
    k_proj: str = 'k_proj.'
    v_proj: str = 'v_proj.'
    out_proj: str = 'o_proj.'
    use_merged_qkv: bool = False

@dataclass
class MLASelfAttnKeyMapping:
    # NOTE: currently not used
    pass

@dataclass
class MLPKeyMapping:
    gate_proj: str = 'gate_proj.'
    up_proj: str = 'up_proj.'
    down_proj: str = 'down_proj.'
    gate_up_proj: str = 'gate_up_proj.'
    use_merged_gate_up: bool = False

@dataclass
class MoELayerKeyMapping:
    # NOTE: currently not used
    pass


@dataclass
class DecoderLayerKeyMapping:
    input_layernorm: str = 'input_layernorm.' # if is_vision_block, norm1.
    self_attn: str = 'self_attn.'
    self_attn_cfg: Union[SelfAttnKeyMapping, MLASelfAttnKeyMapping] = field(default=SelfAttnKeyMapping)
    pre_mlp_layernorm: str = 'post_attention_layernorm.'  # if is_vision_block, norm2.
    mlp: str = 'mlp.'
    mlp_cfg: Union[MLPKeyMapping, MoELayerKeyMapping] = field(default=MLPKeyMapping)


@dataclass
class LanguageModelKeyMapping:
    word_embeddings: str = 'model.'
    decoder_layer: str = 'model.layers.'
    decoder_layer_cfg: DecoderLayerKeyMapping = field(default=DecoderLayerKeyMapping)
    final_layernorm: str = 'model.norm.'
    output_layer: str = ''
