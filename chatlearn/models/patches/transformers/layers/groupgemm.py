"""groupgemm layer with transformer_engine ops"""
from transformers.activations import ACT2FN
import torch
from torch import nn

try:
    from transformer_engine.pytorch.cpp_extensions import grouped_gemm
except ImportError:
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm as grouped_gemm
from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace
from transformer_engine.pytorch.permutation import (
    moe_permute,
    moe_unpermute,
)

from chatlearn.utils import is_te_min_version

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
            grouped_input, row_id_map = moe_permute(hidden_states, selected_experts.to(torch.int32), map_type='index')
        else:
            grouped_input, row_id_map = moe_permute(hidden_states, selected_experts.to(torch.int32))
        probs = router_weights.T.contiguous().view(-1, 1).to(torch.float32)

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
