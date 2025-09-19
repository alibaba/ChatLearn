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
"""patches for qwen3-moe model"""
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn
import torch.nn.functional as F

from chatlearn.models.patches.transformers.layers.groupgemm import MoeGroupMLP

class Qwen3MoeSparseMoeBlock_Grouped(nn.Module):
    """ MOE Block support grouped linear """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.group_mlp = MoeGroupMLP(config, config.moe_intermediate_size)

    def topk_expert(self, logits):
        routing_weights = F.softmax(logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        topk_map = torch.zeros_like(logits).int().scatter(1, selected_experts, 1).bool()
        tokens_per_expert = topk_map.sum(dim=0)
        return routing_weights, selected_experts, tokens_per_expert

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        ori_shape = (batch_size, sequence_length, hidden_dim)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts, tokens_per_expert = self.topk_expert(router_logits)

        final_hidden_states = self.group_mlp(
                                hidden_states,
                                routing_weights.to(hidden_states.dtype),
                                selected_experts,
                                tokens_per_expert
                                )
        final_hidden_states = final_hidden_states.view(ori_shape)
        return final_hidden_states, router_logits

def apply_group_gemm_patch(model):
    cnt = 0
    with torch.device('meta'):
        dummy_moe_layer = Qwen3MoeSparseMoeBlock_Grouped(model.config)
    num_experts = model.config.num_experts
    size_0 = dummy_moe_layer.group_mlp.gate_proj.weight.shape[0] // num_experts
    size_1 = dummy_moe_layer.group_mlp.down_proj.weight.shape[0] // num_experts

    def copy_expert_weights(i, moe_group_layer, layer, size_0, size_1):
        start_idx_0 = i * size_0
        end_idx_0 = (i + 1) * size_0
        start_idx_1 = i * size_1
        end_idx_1 = (i + 1) * size_1

        moe_group_layer.group_mlp.gate_proj.weight.data[start_idx_0:end_idx_0].copy_(
            layer.mlp.experts[i].gate_proj.weight.data
        )
        moe_group_layer.group_mlp.up_proj.weight.data[start_idx_0:end_idx_0].copy_(
            layer.mlp.experts[i].up_proj.weight.data
        )
        moe_group_layer.group_mlp.down_proj.weight.data[start_idx_1:end_idx_1].copy_(
            layer.mlp.experts[i].down_proj.weight.data
        )

    for layer in model.model.layers:
        cnt += 1
        if model.device.type == 'meta':
            with torch.device('meta'):
                moe_group_layer = Qwen3MoeSparseMoeBlock_Grouped(model.config).to(model.dtype)
        else:
            moe_group_layer = Qwen3MoeSparseMoeBlock_Grouped(model.config).to(model.dtype)
            moe_group_layer.gate.weight.data.copy_(layer.mlp.gate.weight.data)
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [
                    executor.submit(
                        copy_expert_weights,
                        i, moe_group_layer, layer, size_0, size_1
                    )
                    for i in range(num_experts)
                ]
                for future in futures:
                    future.result()
        old_mlp = layer.mlp
        del old_mlp
        layer.register_module("mlp",  moe_group_layer)
