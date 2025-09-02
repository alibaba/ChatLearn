# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
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
"""patches for qwen3 model"""
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist

from transformers.cache_utils import Cache
from transformers.models.qwen3 import modeling_qwen3
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from chatlearn.utils.communication_op import all_to_all, get_sp_parallel_group

def qwen3_sp_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    sp_group = get_sp_parallel_group()
    if sp_group is not None:
        sp_size = dist.get_world_size(group=sp_group)
        # Split on hidden_size dim, gather on sequence dim
        if sp_size > self.config.num_key_value_heads:
            # Deal with sp_size > num_key_value_head
            # Avoid each rank get incompatible head
            repeat_times = sp_size // self.config.num_key_value_heads
            key_states = repeat_kv(key_states, repeat_times)
            value_states = repeat_kv(value_states, repeat_times)
            # Change self.num_key_value_groups, eager_attention_forward use this value to
            # determine how many times kv will be repeated
            self.num_key_value_groups = self.config.num_attention_heads // sp_size
        query_states = all_to_all(input_tensor=query_states, sp_group=sp_group, split_dim=1, gather_dim=2)
        key_states = all_to_all(input_tensor=key_states, sp_group=sp_group, split_dim=1, gather_dim=2)
        value_states = all_to_all(input_tensor=value_states, sp_group=sp_group, split_dim=1, gather_dim=2)

    cos, sin = position_embeddings
    query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        assert sp_group is None, "Sp is not supported for transformers generation"
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = modeling_qwen3.eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # diff with Llama
        **kwargs,
    )

    if sp_group is not None:
        sp_size = dist.get_world_size(group=sp_group)
        # Merge heads
        attn_output = attn_output.reshape(input_shape[0], input_shape[1] * sp_size, -1).contiguous()
        # For attn_output, split on sequence dim, gather on hidden_size dim
        attn_output = all_to_all(input_tensor=attn_output, sp_group=sp_group, split_dim=1, gather_dim=2)
        # For attn_weights, split on sequence dim, gather on head
    else:
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

# TODO: version control
def register_sp_attention_forward():
    modeling_qwen3.Qwen3Attention.forward = qwen3_sp_attention_forward
