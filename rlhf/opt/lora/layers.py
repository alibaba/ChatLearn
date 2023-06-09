# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
"""lora layers."""

import os
import math
from typing import Optional
import ray
import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F
import torch.nn.init as init
from .utils import recursive_getattr, recursive_setattr
from torch.cuda.amp import custom_fwd, custom_bwd


from megatron import get_args
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_global_memory_buffer,
)
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    LinearWithGradAccumulationAndAsyncCommunication,
    RowParallelLinear,
    VocabParallelEmbedding,
    _initialize_affine_weight_gpu,
    linear_with_grad_accumulation_and_async_allreduce,
)

from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.utils import VocabUtility

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

from rlhf import get_args as get_rlhf_args
from rlhf.utils.arguments import RLHFConfig
from rlhf.utils.constant import LORA_LAYER, QKV_LAYER_NAME
from rlhf.utils.global_vars import is_initialized


class LinearWithGradAccumulationAndAsyncCommunication_LoRA(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce_LoRA"""

    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion,
                async_grad_allreduce, sequence_parallel):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel

        if sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group())
            total_input = all_gather_buffer
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        if ctx.sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group(), async_op=True)

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        else:
            total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel:
            handle.wait()

        # Doing gather + slicing during the NeMo forward pass can make this tensor 
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only 
        # clones it if it's not contiguous: 
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1],
                                       grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1],
				       total_input.shape[2])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                    grad_input, group=get_tensor_model_parallel_group(), async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(dim_size, dtype=input.dtype,
                                         device=torch.cuda.current_device(),
                                         requires_grad=False)
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input,
                                                            group=get_tensor_model_parallel_group(),
                                                            async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation


        if weight.requires_grad:
            if ctx.gradient_accumulation_fusion:
                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
                elif weight.main_grad.dtype == torch.float16:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, weight.main_grad)
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
                grad_weight = None
            else:
                grad_weight = grad_output.t().matmul(total_input)
        else:
            grad_weight = None
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


def linear_with_grad_accumulation_and_async_allreduce_LoRA(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Arguments:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): Perform the gradient
        accumulation fusion, requires the custom CUDA extension
        fused_weight_gradient_mlp_cuda module. To use
        gradient_accumulation_fusion you must install APEX with
        --cpp_ext and --cuda_ext. For example: "pip install
        --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
        " Note that the extension requires CUDA>=11. Otherwise, you
        must turn off gradient accumulation fusion."

    async_grad_allreduce (bool required): Do the allreduce of input
        gradients asyncronously with the computation of weight
        gradients. If sequence_parallel_enabled is True, this must be
        False, as no all reduce is performed.

    sequence_parallel_enabled (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.
    """
    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel_enabled,
    ]

    if not linear_with_grad_accumulation_and_async_allreduce_LoRA.warned:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel_enabled:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup")
                linear_with_grad_accumulation_and_async_allreduce_LoRA.warned = True

            if async_grad_allreduce:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup")
                linear_with_grad_accumulation_and_async_allreduce_LoRA.warned = True

    return LinearWithGradAccumulationAndAsyncCommunication_LoRA.apply(*args)

linear_with_grad_accumulation_and_async_allreduce_LoRA.warned = False
linear_with_grad_accumulation_and_async_allreduce = linear_with_grad_accumulation_and_async_allreduce_LoRA
LinearWithGradAccumulationAndAsyncCommunication.backward = LinearWithGradAccumulationAndAsyncCommunication_LoRA.backward

class ColumnParallelLinear_LoRA(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(self, weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,
                 bias=None,
                 **kwargs):
        super(ColumnParallelLinear_LoRA, self).__init__()

        # Keep input parameters
        self.gather_output = kwargs.get("gather_output", True)
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()

        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = kwargs.get("skip_bias_add", False)
        input_size = kwargs.get("input_size")
        output_size = kwargs.get("output_size")

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        self.async_tensor_model_parallel_allreduce = (
                args.async_tensor_model_parallel_allreduce and
                world_size > 1)
        self.sequence_parallel = (
                args.sequence_parallel and
                world_size > 1)
        assert not self.async_tensor_model_parallel_allreduce or \
            not self.sequence_parallel
        self.gradient_accumulation_fusion = args.gradient_accumulation_fusion

        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            lora_dim, columns
            ))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(rows, lora_dim))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight, self.lora_right_weight)
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight, self.lora_right_weight)
        self.fuse_lora = False

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce or \
                self.sequence_parallel:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce_LoRA(
            input=input_parallel,
            weight=self.weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel,
        )
        residual = linear_with_grad_accumulation_and_async_allreduce_LoRA(
            input=input_parallel,
            weight=self.lora_right_weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel,
        )
        residual = linear_with_grad_accumulation_and_async_allreduce_LoRA(
            input=residual,
            weight=self.lora_left_weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=False,
        )
        residual = self.lora_dropout(residual)
        output_parallel = output_parallel + self.lora_scaling * residual

        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear_LoRA(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
    """

    def __init__(self, weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,
                 bias=None,
                 **kwargs):
        super(RowParallelLinear_LoRA, self).__init__()

        self.input_is_parallel = kwargs.get("input_is_parallel", False)
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = kwargs.get("skip_bias_add", False)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        self.gradient_accumulation_fusion = args.gradient_accumulation_fusion

        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            lora_dim, columns
            ))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(rows, lora_dim))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight, self.lora_right_weight)
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight, self.lora_right_weight)
        self.fuse_lora = False

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce_LoRA(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
        )
        residual = linear_with_grad_accumulation_and_async_allreduce_LoRA(
            input=input_parallel,
            weight=self.lora_right_weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
        )
        residual = linear_with_grad_accumulation_and_async_allreduce_LoRA(
            input=residual,
            weight=self.lora_left_weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
        )
        residual = self.lora_dropout(residual)

        output_parallel = output_parallel + self.lora_scaling * residual

        # All-reduce across all the partitions.
        if self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias


class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,
                 bias=None,
                 **kwargs):
        super(LinearLayer_LoRA, self).__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            columns,
            lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight, self.lora_right_weight)
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight, self.lora_right_weight)
        self.fuse_lora = False

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(
                input, self.weight,
                self.bias) + (self.lora_dropout(input) @ self.lora_right_weight
                              @ self.lora_left_weight) * self.lora_scaling


class VocabParallelEmbedding_LoRA(nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,
                 bias=None,
                 **kwargs):
        super(VocabParallelEmbedding_LoRA, self).__init__()
        # Set the detauls for compatibility.
        self.padding_idx = kwargs.get("padding_idx", None)
        self.max_norm = kwargs.get("max_norm", None)
        self.norm_type = kwargs.get("norm_type", 2.)
        self.scale_grad_by_freq = kwargs.get("scale_grad_by_freq", False)
        self.sparse = kwargs.get("sparse", False)
        self.num_embeddings = kwargs.get("num_embeddings")
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)

        # Allocate weights and initialize.
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            columns,
            lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = False

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        if not self.fuse_lora:
            after_A = F.embedding(
                    masked_input, self.lora_left_weight.T, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            output_parallel += (after_A @ self.lora_right_weight.T) * self.lora_scaling

        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class Embedding_LoRA(nn.Module):
    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,
                 bias=None,
                 **kwargs):
        super(Embedding_LoRA, self).__init__()
        self.padding_idx = kwargs.get("padding_idx", None)
        self.max_norm = kwargs.get("max_norm", None)
        self.norm_type = kwargs.get("norm_type", 2.)
        self.scale_grad_by_freq = kwargs.get("scale_grad_by_freq", False)
        self.sparse = kwargs.get("sparse", False)
        self.num_embeddings = kwargs.get("num_embeddings")
        # Set the detauls for compatibility.
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            columns,
            lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        self.weight.shared = True
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = False

    def forward(self, input_):
        output = F.embedding(
            input_, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        if not self.fuse_lora:
            after_A = F.embedding(
                    input_, self.lora_left_weight.T, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            output += (after_A @ self.lora_right_weight.T) * self.lora_scaling    
        return output



ALL_LORA_LAYER = (
    ColumnParallelLinear_LoRA,
    Embedding_LoRA,
    LinearLayer_LoRA,
    RowParallelLinear_LoRA,
    VocabParallelEmbedding_LoRA
)

LORA_LAYER_MAP = {
    "ColumnParallelLinear": ColumnParallelLinear_LoRA,
    "Embedding": Embedding_LoRA,
    "LinearLayer": LinearLayer_LoRA,
    "RowParallelLinear": RowParallelLinear_LoRA,
    "VocabParallelEmbedding": VocabParallelEmbedding_LoRA
}


# convert layer to LoRA
def convert_layer_to_lora(model,
                          part_module_name=None,
                          lora_dim=None,
                          lora_scaling=None,
                          lora_dropout=None,
                          lora_layer=None,
                          column_only_qkv=None):

    if is_initialized():
        func = get_rlhf_args().rlhf_args
    else:
        func = RLHFConfig
    rlhf_part_module_name = func.part_module_name
    rlhf_lora_dim = func.lora_dim
    rlhf_lora_scaling = func.lora_scaling
    rlhf_lora_dropout = func.lora_dropout
    rlhf_lora_layer = func.lora_layer
    rlhf_column_only_qkv = func.column_only_qkv

    part_module_name = part_module_name if part_module_name is not None else rlhf_part_module_name
    lora_dim = lora_dim if lora_dim is not None else rlhf_lora_dim
    lora_scaling = lora_scaling if lora_scaling is not None else rlhf_lora_scaling
    lora_dropout = lora_dropout if lora_dropout is not None else rlhf_lora_dropout
    layers_to_convert = lora_layer if lora_layer is not None else rlhf_lora_layer
    column_only_qkv = column_only_qkv if column_only_qkv is not None else rlhf_lora_layer

    if lora_dim <= 0:
        return model

    layers_to_convert = layers_to_convert.split(",")
    assert all([layer in LORA_LAYER_MAP for layer in layers_to_convert]), \
        "Unsupport layer to enable lora, {}. Only support {} for now.".format(layers_to_convert, DEFAULT_LORA_LAYER)

    repalce_name = {}
    for name, module in model.named_modules():
        if part_module_name is not None and part_module_name not in name:
            continue
        if isinstance(module, nn.Linear) and "Linear" in layers_to_convert:
            repalce_name[name] = LinearLayer_LoRA
        elif isinstance(module, RowParallelLinear) and "RowParallelLinear" in layers_to_convert:
            repalce_name[name] = RowParallelLinear_LoRA
        elif isinstance(module, ColumnParallelLinear) and "ColumnParallelLinear" in layers_to_convert:
            if column_only_qkv and any([ele not in name for ele in QKV_LAYER_NAME]):
                continue
            repalce_name[name] = ColumnParallelLinear_LoRA
        elif isinstance(module, VocabParallelEmbedding) and "VocabParallelEmbedding" in layers_to_convert:
            repalce_name[name] = VocabParallelEmbedding_LoRA
        elif isinstance(module, Embedding) and "Embedding" in layers_to_convert:
            repalce_name[name] = Embedding_LoRA
        else:
            pass
        
    for name, func in repalce_name.items():
        module = recursive_getattr(model, name)
        kwargs = dict()
        if hasattr(module, "input_is_parallel"):
            kwargs["input_is_parallel"] = module.input_is_parallel
        if hasattr(module, "skip_bias_add"):
            kwargs["skip_bias_add"] = module.skip_bias_add
        if hasattr(module, "gather_output"):
            kwargs["gather_output"] = module.gather_output
        if hasattr(module, "input_size"):
            kwargs["input_size"] = module.input_size
        if hasattr(module, "output_size"):
            kwargs["output_size"] = module.output_size
        if hasattr(module, "padding_idx"):
            kwargs["padding_idx"] = module.padding_idx
        if hasattr(module, "max_norm"):
            kwargs["max_norm"] = module.max_norm
        if hasattr(module, "norm_type"):
            kwargs["norm_type"] = module.norm_type
        if hasattr(module, "scale_grad_by_freq"):
            kwargs["scale_grad_by_freq"] = module.scale_grad_by_freq
        if hasattr(module, "sparse"):
            kwargs["sparse"] = module.sparse
        if hasattr(module, "num_embeddings"):
            kwargs["num_embeddings"] = module.num_embeddings
        tmp = func(
            module.weight, lora_dim, lora_scaling, lora_dropout,
            module.bias if hasattr(module, "bias") else None, **kwargs).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)

    only_optimize_lora_parameters(model)

    return model


def fuse_lora_layer(model):
    if isinstance(model, list):
        model = model[0]
    for name, module in model.named_modules():
        if isinstance(module, ALL_LORA_LAYER):
            module.fuse_lora_weight()

def unfuse_lora_layer(model):
    if isinstance(model, list):
        model = model[0]
    for name, module in model.named_modules():
        if isinstance(module, ALL_LORA_LAYER):
            module.unfuse_lora_weight()


def only_optimize_lora_parameters(model, excluded_flags=["bias"], excluded_attrs=["sequence_parallel"], is_training=True):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name or \
                any([getattr(param, ele, False) for ele in excluded_attrs]) or \
                any([ele in name for ele in excluded_flags]):
            param.requires_grad = is_training
        else:
            param.requires_grad = False

    print_trainable_parameters(model)

    return model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    if torch.distributed.get_rank() == 0:
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
