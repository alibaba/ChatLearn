"""patches for qwen3 vl model"""
from typing import Optional
import torch

def Qwen3VLBlock_patched_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
    # =========================================================================
    # add force dype change for qwen3_vl or backward will occur type error
    hidden_states = hidden_states.to(self.norm1.weight.dtype)
    # =========================================================================

    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states

def apply_qwen3vl_patch():
    # pylint: disable=import-outside-toplevel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeVisionBlock
    Qwen3VLVisionBlock.forward = Qwen3VLBlock_patched_forward
    Qwen3VLMoeVisionBlock.forward = Qwen3VLBlock_patched_forward
