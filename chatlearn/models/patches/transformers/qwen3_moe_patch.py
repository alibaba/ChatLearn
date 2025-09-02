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

try:
    from transformer_engine.pytorch.cpp_extensions import grouped_gemm
except ImportError:
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm as grouped_gemm
from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace
from transformer_engine.pytorch.permutation import (
    moe_permute,
    moe_unpermute,
)
from transformers.activations import ACT2FN

from chatlearn.utils import is_te_min_version


class GroupGemm(torch.autograd.Function):
    """ Autograd function for grouped gemm"""
    @staticmethod
    def forward(
        ctx,
        inp,
        m_splits,
        use_bias,
        is_grad_enabled,
        activation_dtype,
        *weights_bias
    ) -> torch.Tensor:
        n_gemm = len(m_splits)
        weights = weights_bias[:n_gemm]
        bias = weights_bias[n_gemm:]
        in_features = weights[0].shape[-1]
        inputmats = torch.split(inp.view(-1, in_features), m_splits)
        output_tensor = torch.empty(
            [sum(m_splits), weights[0].shape[0]],
            dtype=activation_dtype,
            device=inputmats[0].device,
        )
        grouped_gemm_kwargs = {'dtype': activation_dtype}
        if is_te_min_version("2.0.0"):
            grouped_gemm_kwargs = {'out_dtype': activation_dtype, 'm_splits': m_splits}
        _ = grouped_gemm(
            A=weights,
            B=inputmats,
            out=torch.split(output_tensor, m_splits),
            workspaces=get_multi_stream_cublas_workspace(),
            bias=bias,
            use_bias=use_bias,
            **grouped_gemm_kwargs
        )
        if is_grad_enabled:
            ctx.save_for_backward(
                *inputmats,
                *weights,
            )
            ctx.m_splits = m_splits
            ctx.num_gemm = n_gemm
            ctx.activation_dtype = activation_dtype
            ctx.use_bias = use_bias
            ctx.inp_shape = inp.shape
        return output_tensor.view(-1, *inp.shape[1:-1], output_tensor.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        with torch.cuda.nvtx.range("_GroupedLinear_backward"):
            saved_tensors = ctx.saved_tensors
            inputmats = saved_tensors[:ctx.num_gemm]
            weights = saved_tensors[ctx.num_gemm:]

            grad_output = grad_output.contiguous()
            grad_output_mats = torch.split(
                grad_output.view(-1, grad_output.shape[-1]), ctx.m_splits
            )
            #dgrad
            dgrad = torch.empty(
                (sum(ctx.m_splits), weights[0].size(1)),
                dtype=ctx.activation_dtype,
                device=grad_output.device,
            )
            grouped_gemm_kwargs = {'dtype': ctx.activation_dtype}
            if is_te_min_version("2.0.0"):
                grouped_gemm_kwargs = {'out_dtype': ctx.activation_dtype, 'm_splits': ctx.m_splits}
            grouped_gemm(
                A=weights,
                B=grad_output_mats,
                out=torch.split(dgrad, ctx.m_splits),
                workspaces=get_multi_stream_cublas_workspace(),
                layout="NN",
                grad=True,
                **grouped_gemm_kwargs
            )

            #wgrad
            wgrad_list = [
                torch.empty(w.size(), dtype=ctx.activation_dtype, device=w.device)
                for w in weights
            ]
            _, grad_biases, _ = grouped_gemm(
                A=inputmats,
                B=grad_output_mats,
                out=wgrad_list,
                workspaces=get_multi_stream_cublas_workspace(),
                layout="NT",
                grad=True,
                use_bias=ctx.use_bias,
                **grouped_gemm_kwargs
            )
            if not ctx.use_bias:
                grad_biases = [None]
            return (
                dgrad.view(ctx.inp_shape),
                None,
                None,
                None,
                None,
                *wgrad_list,
                *grad_biases,
            )

def grouped_linear(inp, m_splits, use_bias, is_grad_enabled, activation_dtype, weights_bias, num_experts):
    weights_bias = torch.chunk(weights_bias, num_experts, dim=0)
    output = GroupGemm.apply(
        inp,
        m_splits,
        use_bias,
        is_grad_enabled,
        activation_dtype,
        *weights_bias
    )
    return output


class Linear(nn.Module):
    """used for empty init gate_proj,up_proj,down_proj"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)


class MoeGroupMLP(nn.Module):
    """ Group MLP Layer """
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.num_experts = config.num_experts
        hidden_size = config.hidden_size

        self.gate_proj = Linear(hidden_size, intermediate_size * self.num_experts, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size * self.num_experts, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size * self.num_experts, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, router_weights, selected_experts, token_per_expert) -> torch.Tensor:
        flat_seqlen = hidden_states.shape[0]
        topk = router_weights.shape[1]
        if is_te_min_version("2.1.0"):
            grouped_input, row_id_map = moe_permute(hidden_states, selected_experts, map_type='index')
        else:
            grouped_input, row_id_map = moe_permute(hidden_states, selected_experts)
        probs = router_weights.T.contiguous().view(-1, 1)

        token_per_expert = token_per_expert.tolist()
        gate_output = self.act_fn(grouped_linear(
            inp = grouped_input,
            m_splits = token_per_expert,
            use_bias = False,
            is_grad_enabled = self.training,
            activation_dtype = hidden_states.dtype,
            weights_bias = self.gate_proj.weight,
            num_experts=self.num_experts
        ))
        up_output = grouped_linear(
            inp = grouped_input,
            m_splits = token_per_expert,
            use_bias = False,
            is_grad_enabled = self.training,
            activation_dtype = hidden_states.dtype,
            weights_bias = self.up_proj.weight,
            num_experts=self.num_experts
        ) * gate_output
        down_output = grouped_linear(
            inp = up_output,
            m_splits = token_per_expert,
            use_bias = False,
            is_grad_enabled = self.training,
            activation_dtype = hidden_states.dtype,
            weights_bias = self.down_proj.weight,
            num_experts=self.num_experts
        )
        if is_te_min_version("2.1.0"):
            final_hidden_states = moe_unpermute(down_output, row_id_map, probs, map_type='index')
        else:
            final_hidden_states = moe_unpermute(down_output, row_id_map, probs)
        final_hidden_states = final_hidden_states.view(topk, flat_seqlen, -1).permute(1,0,2)
        final_hidden_states = torch.sum(final_hidden_states, dim=1).squeeze(1)
        return final_hidden_states

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
