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
"""lora layers."""

import math

import importlib.util
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Embedding
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from chatlearn.models.megatron.lora.initializer import distributed_kaiming_uniform_
from chatlearn.models.megatron.lora.utils import recursive_getattr, recursive_setattr
from chatlearn.utils.arguments import LoraConfig
from chatlearn.utils.constant import LORA_WEIGHT_PREFIX
from chatlearn.utils.constant import QKV_LAYER_NAME
from chatlearn.utils.global_vars import get_args as get_runtime_args
from chatlearn.utils.global_vars import is_initialized

megatron_exist = importlib.util.find_spec("megatron")
if megatron_exist:
    from chatlearn.utils.megatron_import_helper import get_args
    from chatlearn.utils.megatron_import_helper import mpu
    from chatlearn.utils.megatron_import_helper import Float16Module
    from chatlearn.utils.megatron_import_helper import MegatronOptimizer
    from chatlearn.utils.megatron_import_helper import unwrap_model
    from chatlearn.utils.megatron_import_helper import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size
    )
    from chatlearn.utils.megatron_import_helper import (  # pylint: disable=unused-import
        ColumnParallelLinear,
        linear_with_frozen_weight,
        linear_with_grad_accumulation_and_async_allreduce,
        LinearWithGradAccumulationAndAsyncCommunication,
        RowParallelLinear,
        VocabParallelEmbedding
    )

    from chatlearn.utils.megatron_import_helper import (
        copy_to_tensor_model_parallel_region,
        gather_from_tensor_model_parallel_region,
        reduce_from_tensor_model_parallel_region,
        scatter_to_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region
    )
    from chatlearn.utils.megatron_import_helper import VocabUtility


class LoraBase(torch.nn.Module):  # pylint: disable=abstract-method
    """Lora Base"""

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.fuse_lora:
            sd = {key: value for key, value in sd.items() if not key.startswith(LORA_WEIGHT_PREFIX)}
        return sd


class ColumnParallelLinear_LoRA(LoraBase):
    """LoRA version of megatron.core.tensor_parallel.layers.ColumnParallelLinear.

    Arguments:
        weight: weight of original ColumnParallelLinear module.
        lora_dim: lora rank dim.
        lora_scaling: lora scaling value.
        bias: bias of original ColumnParallelLinear module.
        kwargs: args of original ColumnParallelLinear module.
    """

    def __init__(self, weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,
                 bias=None,
                 **kwargs):
        super().__init__()

        # Keep input parameters
        self.gather_output = kwargs.get("gather_output", True)
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()

        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = kwargs.get("skip_bias_add", False)

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
        self.fan_in = columns
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
        distributed_kaiming_uniform_(self.lora_right_weight, self.fan_in, a=math.sqrt(5))
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
        output_parallel = linear_with_frozen_weight(
            input=input_parallel,
            weight=self.weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel=self.sequence_parallel,
        )
        residual = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.lora_right_weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel=self.sequence_parallel,
        )
        residual = linear_with_grad_accumulation_and_async_allreduce(
            input=residual,
            weight=self.lora_left_weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel=False,
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


class RowParallelLinear_LoRA(LoraBase):
    """LoRA version of megatron.core.tensor_parallel.layers.RowParallelLinear.

    Arguments:
        weight: weight of original RowParallelLinear module.
        lora_dim: lora rank dim.
        lora_scaling: lora scaling value.
        bias: bias of original RowParallelLinear module.
        kwargs: args of original RowParallelLinear module.
    """

    def __init__(self, weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,
                 bias=None,
                 **kwargs):
        super().__init__()

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
        self.fan_in = columns
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
        distributed_kaiming_uniform_(self.lora_right_weight, self.fan_in, a=math.sqrt(5))
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
        output_parallel = linear_with_frozen_weight(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel=False,
        )
        residual = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.lora_right_weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel=False,
        )
        residual = linear_with_grad_accumulation_and_async_allreduce(
            input=residual,
            weight=self.lora_left_weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel=False,
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


class LinearLayer_LoRA(LoraBase):
    """LoRA version of torch.nn.Linear.

    Arguments:
        weight: weight of original torch.nn.Linear module.
        lora_dim: lora rank dim.
        lora_scaling: lora scaling value.
        bias: bias of original torch.nn.Linear module.
        kwargs: args of original torch.nn.Linear module.
    """

    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,
                 bias=None):
        super().__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:  # pylint: disable=bare-except
            rows, columns = weight.shape
        self.fan_in = columns
        self.lora_right_weight = nn.Parameter(torch.zeros(
            lora_dim, columns))  # apply transpose so in forward we do not need to
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
        distributed_kaiming_uniform_(self.lora_right_weight, self.fan_in, a=math.sqrt(5))
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

    def forward(self, inputs):
        if self.fuse_lora:
            return F.linear(inputs, self.weight, self.bias)
        else:
            return F.linear(
                inputs, self.weight,
                self.bias) + (self.lora_dropout(inputs) @ self.lora_right_weight.t()
                              @ self.lora_left_weight.t()) * self.lora_scaling


class VocabParallelEmbedding_LoRA(LoraBase):
    """LoRA version of megatron.core.tensor_parallel.layers.VocabParallelEmbedding.

    Arguments:
        weight: weight of original VocabParallelEmbedding module.
        lora_dim: lora rank dim.
        lora_scaling: lora scaling value.
        bias: bias of original VocabParallelEmbedding module.
        kwargs: args of original VocabParallelEmbedding module.
    """

    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,
                 bias=None,
                 **kwargs):
        super().__init__()
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
        self.fan_in = columns
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
        distributed_kaiming_uniform_(self.lora_right_weight, self.fan_in, a=math.sqrt(5))
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


class Embedding_LoRA(LoraBase):
    """LoRA version of torch.nn.Embedding.

    Arguments:
        weight: weight of original torch.nn.Embedding module.
        lora_dim: lora rank dim.
        lora_scaling: lora scaling value.
        bias: bias of original torch.nn.Embedding module.
        kwargs: args of original torch.nn.Embedding module.
    """

    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,
                 bias=None,
                 **kwargs):
        super().__init__()
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
        self.fan_in = columns
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
        distributed_kaiming_uniform_(self.lora_right_weight, self.fan_in, a=math.sqrt(5))
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


class MegatronOptimizer_LoRA(MegatronOptimizer):
    """
    MegatronOptimizer for LoRA
    """
    def allreduce_word_embedding_grads(self, args):
        """
        All-reduce word embedding grads.
        Reduce grads across first and last stages to ensure that word_embeddings
        parameters stay in sync. This should only run for models that support
        pipelined model parallelism (BERT and GPT-2).
        """
        if mpu.is_rank_in_embedding_group(ignore_virtual=True) and \
            mpu.get_pipeline_model_parallel_world_size() > 1:
            if mpu.is_pipeline_first_stage(ignore_virtual=True):
                unwrapped_model = self.models[0]
            elif mpu.is_pipeline_last_stage(ignore_virtual=True):
                unwrapped_model = self.models[-1]
            else:  # We do not support the interleaved schedule for T5 yet.
                unwrapped_model = self.models[0]

            if hasattr(unwrapped_model, "share_word_embeddings"):
                from chatlearn.utils.megatron_import_helper import DistributedDataParallel as LocalDDP # pylint: disable=import-outside-toplevel
                unwrapped_model = unwrap_model(
                    unwrapped_model, (torchDDP, LocalDDP, Float16Module))
                if unwrapped_model.share_word_embeddings:
                    word_embeddings_weight = unwrapped_model.word_embeddings_weight()
                    if word_embeddings_weight.requires_grad:
                        if args.DDP_impl == 'local':
                            grad = word_embeddings_weight.main_grad
                        else:
                            grad = word_embeddings_weight.grad
                        torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())
            elif hasattr(unwrapped_model, "share_embeddings_and_output_weights"):
                unwrapped_model = unwrap_model(unwrapped_model)
                if unwrapped_model.share_embeddings_and_output_weights:
                    weight = unwrapped_model.shared_embedding_or_output_weight()
                    if weight.requires_grad:
                        grad = weight.main_grad
                        torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())


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
        default_args = get_runtime_args().active_module_args.lora
    else:
        default_args = LoraConfig

    part_module_name = part_module_name if part_module_name is not None else default_args.part_module_name
    lora_dim = lora_dim if lora_dim is not None else default_args.lora_dim
    lora_scaling = lora_scaling if lora_scaling is not None else default_args.lora_scaling
    lora_dropout = lora_dropout if lora_dropout is not None else default_args.lora_dropout
    layers_to_convert = lora_layer if lora_layer is not None else default_args.lora_layer
    column_only_qkv = column_only_qkv if column_only_qkv is not None else default_args.column_only_qkv

    if lora_dim <= 0:
        return model

    layers_to_convert = layers_to_convert.split(",")
    assert all(layer in LORA_LAYER_MAP for layer in layers_to_convert), \
        "Unsupport layer to enable lora, {}. Only support {} for now.".format(layers_to_convert, ALL_LORA_LAYER)

    MegatronOptimizer.allreduce_word_embedding_grads = MegatronOptimizer_LoRA.allreduce_word_embedding_grads

    repalce_name = {}
    for name, module in model.named_modules():
        if part_module_name is not None and part_module_name not in name:
            continue
        if isinstance(module, nn.Linear) and "LinearLayer" in layers_to_convert:
            repalce_name[name] = LinearLayer_LoRA
        elif isinstance(module, RowParallelLinear) and "RowParallelLinear" in layers_to_convert:
            repalce_name[name] = RowParallelLinear_LoRA
        elif isinstance(module, ColumnParallelLinear) and "ColumnParallelLinear" in layers_to_convert:
            if column_only_qkv and any(ele not in name for ele in QKV_LAYER_NAME):
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
        kwargs = {}
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
    for _, module in model.named_modules():
        if isinstance(module, ALL_LORA_LAYER):
            module.fuse_lora_weight()


def unfuse_lora_layer(model):
    if isinstance(model, list):
        model = model[0]
    for _, module in model.named_modules():
        if isinstance(module, ALL_LORA_LAYER):
            module.unfuse_lora_weight()


def only_optimize_lora_parameters(model, excluded_flags="bias", excluded_attrs="sequence_parallel", is_training=True):
    # turn off the gradient of all the parameters except the LoRA parameters
    excluded_flags = excluded_flags.split(",")
    excluded_attrs = excluded_attrs.split(",")
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name or \
            any(getattr(param, ele, False) for ele in excluded_attrs) or \
            any(ele in name for ele in excluded_flags):
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
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    if torch.distributed.get_rank() == 0:
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
