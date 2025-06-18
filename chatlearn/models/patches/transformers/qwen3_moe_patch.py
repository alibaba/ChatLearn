import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_engine.pytorch.cpp_extensions import grouped_gemm
from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace
from transformer_engine.pytorch.permutation import (
    moe_permute,
    moe_unpermute,
)
from transformers.activations import ACT2FN
import gc
from concurrent.futures import ThreadPoolExecutor

class GroupGemm(torch.autograd.Function):
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
        #print(bias)
        in_features = weights[0].shape[-1]
        inputmats = torch.split(inp.view(-1, in_features), m_splits)
        output_tensor = torch.empty(
            [sum(m_splits), weights[0].shape[0]],
            dtype=activation_dtype,
            device=inputmats[0].device,
        )
        # print(f"weight dtype: {weights[0].dtype}")
        # print(f"input dtype: {inputmats[0].dtype}")
        _ = grouped_gemm(
            weights,
            inputmats,
            torch.split(output_tensor, m_splits),
            activation_dtype,
            get_multi_stream_cublas_workspace(),
            bias=bias,
            use_bias=use_bias,
        )
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
            grouped_gemm(
                weights,
                grad_output_mats,
                torch.split(dgrad, ctx.m_splits),
                ctx.activation_dtype,
                get_multi_stream_cublas_workspace(),
                layout="NN",
                grad=True,
            )

            #wgrad
            wgrad_list = [
                torch.empty(w.size(), dtype=ctx.activation_dtype, device=w.device)
                for w in weights
            ]
            _, grad_biases, _ = grouped_gemm(
                inputmats,
                grad_output_mats,
                wgrad_list,
                ctx.activation_dtype,
                get_multi_stream_cublas_workspace(),
                layout="NT",
                grad=True,
                use_bias=ctx.use_bias,
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

class MoeGroupMLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.num_experts = config.num_experts
        # gating
        hidden_size = config.hidden_size
        self.gate_weight = nn.Parameter(torch.empty(intermediate_size * self.num_experts, hidden_size))
        self.up_weight = torch.nn.Parameter(torch.empty(intermediate_size * self.num_experts, hidden_size))
        self.down_weight = torch.nn.Parameter(torch.empty(hidden_size * self.num_experts, intermediate_size))
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, hidden_states, router_weights, ori_shape, selected_experts, topk_map, token_per_expert) -> torch.Tensor:
        # expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        batch_size, sequence_length, hidden_dim = ori_shape
        topk = router_weights.shape[1]

        grouped_input, row_id_map = moe_permute(hidden_states, selected_experts)
        probs = router_weights.T.contiguous().view(-1, 1)

        token_per_expert = token_per_expert.tolist()
        gate_output = self.act_fn(grouped_linear(
            inp = grouped_input,
            m_splits = token_per_expert,
            use_bias = False,
            is_grad_enabled = self.training,
            activation_dtype = hidden_states.dtype,
            weights_bias = self.gate_weight,
            num_experts=self.num_experts
        ))
        up_output = grouped_linear(
            inp = grouped_input,
            m_splits = token_per_expert,
            use_bias = False,
            is_grad_enabled = self.training,
            activation_dtype = hidden_states.dtype,
            weights_bias = self.up_weight,
            num_experts=self.num_experts
        ) * gate_output
        down_output = grouped_linear(
            inp = up_output,
            m_splits = token_per_expert,
            use_bias = False,
            is_grad_enabled = self.training,
            activation_dtype = hidden_states.dtype,
            weights_bias = self.down_weight,
            num_experts=self.num_experts
        )
        final_hidden_states = moe_unpermute(down_output, row_id_map, probs)#.reshape(batch_size * sequence_length, 8, hidden_dim)
        final_hidden_states = final_hidden_states.view(topk, sequence_length, -1).permute(1,0,2)
        final_hidden_states = torch.sum(final_hidden_states, dim=1).squeeze(1)
        return final_hidden_states

class Qwen3MoeSparseMoeBlock_Grouped(nn.Module):
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
        topk_masked_gates = torch.zeros_like(logits, dtype=routing_weights.dtype).scatter(1, selected_experts, routing_weights)
        topk_map = torch.zeros_like(logits).int().scatter(1, selected_experts, 1).bool()
        tokens_per_expert = topk_map.sum(dim=0)
        return routing_weights, selected_experts, topk_map, tokens_per_expert
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        ori_shape = (batch_size, sequence_length, hidden_dim)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts, topk_map, tokens_per_expert = self.topk_expert(router_logits)

        final_hidden_states = self.group_mlp(hidden_states, routing_weights.to(hidden_states.dtype), ori_shape, selected_experts, topk_map, tokens_per_expert)
        return final_hidden_states, router_logits

import time

def apply_group_gemm_patch(model):
    cnt = 0
    cpy_cost = []
    with torch.device('meta'):
        dummy_moe_layer = Qwen3MoeSparseMoeBlock_Grouped(model.config)
    num_experts = model.config.num_experts
    size_0 = dummy_moe_layer.group_mlp.gate_weight.shape[0] // num_experts
    size_1 = dummy_moe_layer.group_mlp.down_weight.shape[0] // num_experts

    def copy_expert_weights(i, moe_group_layer, layer, size_0, size_1):
        start_idx_0 = i * size_0
        end_idx_0 = (i + 1) * size_0
        start_idx_1 = i * size_1
        end_idx_1 = (i + 1) * size_1
        
        moe_group_layer.group_mlp.gate_weight.data[start_idx_0:end_idx_0].copy_(
            layer.mlp.experts[i].gate_proj.weight.data
        )
        moe_group_layer.group_mlp.up_weight.data[start_idx_0:end_idx_0].copy_(
            layer.mlp.experts[i].up_proj.weight.data
        )
        moe_group_layer.group_mlp.down_weight.data[start_idx_1:end_idx_1].copy_(
            layer.mlp.experts[i].down_proj.weight.data
        )

    for layer in model.model.layers:
        start = time.time()
        cnt += 1
        if model.device.type == 'meta':
            #print(f"debugyy : {model.device}")
            with torch.device('meta'):
                moe_group_layer = Qwen3MoeSparseMoeBlock_Grouped(model.config).to(model.dtype)
        else:
            #print(f"prepare layer: {time.time() - start}")
            moe_group_layer = Qwen3MoeSparseMoeBlock_Grouped(model.config).to(model.dtype)
            moe_group_layer.gate.weight.data.copy_(layer.mlp.gate.weight.data)
            # for i in range(num_experts):
            #     moe_group_layer.group_mlp.gate_weight.data[i * size_0: (i + 1) * size_0].copy_(layer.mlp.experts[i].gate_proj.weight.data)
            #     moe_group_layer.group_mlp.up_weight.data[i * size_0: (i + 1) * size_0].copy_(layer.mlp.experts[i].up_proj.weight.data)
            #     moe_group_layer.group_mlp.down_weight.data[i * size_1: (i + 1) * size_1].copy_(layer.mlp.experts[i].down_proj.weight.data)
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
        cpy_cost.append(time.time() - start)
        print(f"copy weights: {cpy_cost[-1]}")
        old_mlp = layer.mlp
        del old_mlp
        layer.register_module("mlp",  moe_group_layer)
        print(f"total cost for one layer: {time.time() - start}")
    #gc.collect()
        #print(f"clean and register: {time.time() - start}")
        