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
"""helper to collect shape infos for MCore model"""
from typing import TYPE_CHECKING, Dict
from torch import nn

from .sharded_tensor_info import ShardedTensorInfo

if TYPE_CHECKING:
    from megatron.core.models.gpt import GPTModel

try:
    # pylint: disable=ungrouped-imports
    from transformer_engine.pytorch import RMSNorm, LayerNorm
    from megatron.core import mpu
    from megatron.core.extensions.transformer_engine import (
        TELinear,
        TELayerNormColumnParallelLinear,
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear
    )
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.tensor_parallel import (
        VocabParallelEmbedding,
        ColumnParallelLinear
    )
    HAVE_MEGATRON = True
except ImportError:
    HAVE_MEGATRON = False


def _prepare_metadata(prefix: str, module: nn.Module):
    if not HAVE_MEGATRON:
        raise SystemError("Cannot call this function without megatron")
    results = {}
    if isinstance(module, (RMSNorm, LayerNorm)):
        results['weight'] = ShardedTensorInfo.from_global_shape(
            tuple(module.weight.shape), dtype=module.weight.dtype
        )
        if hasattr(module, 'bias'):
            results['bias'] = ShardedTensorInfo.from_global_shape(
                tuple(module.bias.shape), dtype=module.bias.dtype
            )
    elif isinstance(module, (TELinear, TELayerNormColumnParallelLinear, ColumnParallelLinear)):
        if isinstance(module, ColumnParallelLinear):
            # NOTE: Megatron-LM 0.12.1 still not support `tp_group` attr
            parallel_mode = 'column'
            use_bias = module.bias is not None
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()
        else:
            # NOTE: The parallel_mode of TELinear will be None in below cases:
            # 1. duplicated
            # 2. already handled in megatron: explicit_expert_comm, shared_expert_overlap
            parallel_mode = module.parallel_mode
            if parallel_mode is None:
                if prefix.find('linear_fc1') != -1:
                    parallel_mode = 'column'
                elif prefix.find('linear_fc2') != -1:
                    parallel_mode = 'row'
            use_bias = module.use_bias
            if module.tp_group is None:
                tp_rank, tp_size = 0, 1
                if module.parallel_mode is not None:
                    raise ValueError(f"Suppose module {prefix} to have tp_group initialized!")
                # NOTE: identify `explicit_expert_comm` by module name
                if parallel_mode is not None:
                    # explicit_expert_comm is True, thus in expert
                    tp_rank = mpu.get_expert_tensor_parallel_rank()
                    tp_size = mpu.get_expert_tensor_parallel_world_size()
                # NOTE: otherwise duplicated linear
            else:
                # NOTE: shared_experts have tp_group
                tp_rank, tp_size = module.tp_group.rank(), module.tp_group.size()
        w, h = module.weight.shape
        if parallel_mode == 'row':
            global_shape=(w, h * tp_size)
            axis_fragmentations=(1, tp_size)
            global_offset=(0, tp_rank)
        elif parallel_mode == 'column':
            axis_fragmentations=(tp_size, 1)
            global_offset=(tp_rank, 0)
            global_shape=(w * tp_size, h)
        elif tp_size == 1:
            axis_fragmentations=(1, 1)
            global_offset=(0, 0)
            global_shape=(w, h)
        else:
            if parallel_mode is None:
                parallel_mode = 'none'
            raise ValueError(f"Cannot identify module {prefix}, got parallel mode {parallel_mode} and tp_size {tp_size}")
        results['weight'] = ShardedTensorInfo(
            dtype=module.weight.dtype,
            global_shape=global_shape,
            axis_fragmentations=axis_fragmentations,
            global_offset=global_offset
        )
        if use_bias:
            results['bias'] = ShardedTensorInfo(
                dtype=module.bias.dtype,
                global_shape=global_shape[:1],
                axis_fragmentations=axis_fragmentations[:1],
                global_offset=global_offset[:1]
            )
        layer_norm_weight = getattr(module, 'layer_norm_weight', None)
        if layer_norm_weight is not None:
            results['layer_norm_weight'] = ShardedTensorInfo.from_global_shape(
                layer_norm_weight.shape,
                dtype=layer_norm_weight.dtype
            )
        layer_norm_bias = getattr(module, 'layer_norm_bias', None)
        if layer_norm_bias is not None:
            results['layer_norm_bias'] = ShardedTensorInfo.from_global_shape(
                layer_norm_bias.shape,
                dtype=layer_norm_bias.dtype
            )
    elif isinstance(module, (TEColumnParallelGroupedLinear, TERowParallelGroupedLinear)):
        if module.tp_group is None:
            # explicit_expert_comm is True, thus in expert
            tp_rank = mpu.get_expert_tensor_parallel_rank()
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_rank, tp_size = module.tp_group.rank(), module.tp_group.size()

        for gemm_id in range(module.num_gemms):
            weight = getattr(module, f"weight{gemm_id}")
            w, h = weight.shape
            if isinstance(module, TERowParallelGroupedLinear):
                global_shape=(w, h * tp_size)
                axis_fragmentations=(1, tp_size)
                global_offset=(0, tp_rank)
            else:
                axis_fragmentations=(tp_size, 1)
                global_offset=(tp_rank, 0)
                global_shape=(w * tp_size, h)
            results[f'weight{gemm_id}'] = ShardedTensorInfo(
                dtype=weight.dtype,
                global_shape=global_shape,
                axis_fragmentations=axis_fragmentations,
                global_offset=global_offset
            )
            if module.use_bias:
                bias = getattr(module, f"bias{gemm_id}")
                results[f'bias{gemm_id}'] = ShardedTensorInfo(
                    dtype=bias.dtype,
                    global_shape=global_shape[:1],
                    axis_fragmentations=axis_fragmentations[:1],
                    global_offset=global_offset[:1]
                )
    elif isinstance(module, TopKRouter):
        results['weight'] = ShardedTensorInfo.from_global_shape(
            tuple(module.weight.shape), dtype=module.weight.dtype
        )
        if module.enable_expert_bias:
            results['expert_bias'] = ShardedTensorInfo.from_global_shape(
                tuple(module.expert_bias.shape), dtype=module.expert_bias.dtype
            )
    elif isinstance(module, VocabParallelEmbedding):
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        w, h = module.weight.shape
        results['weight'] = ShardedTensorInfo(
            dtype=module.weight.dtype,
            axis_fragmentations=(tp_size, 1),
            global_offset=(tp_rank, 0),
            global_shape=(w * tp_size, h)
        )
    return results

def build_sharded_info_for_mcore_model(
    model: 'GPTModel'
) -> Dict[str, ShardedTensorInfo]:
    """build sharded tensor info from onloaded GPTModel.

    Args:
        model (GPTModel): The given model

    Returns:
        Dict[str, ShardedTensorInfo]: A dict maps local parameter
        name to sharded_info
    """
    # TODO: can we parse sharded info from sharded_state_dict?
    infos = {}
    for prefix, submodule in model.named_modules():
        if (
            model.share_embeddings_and_output_weights and
            prefix == 'output_layer' and
            model.pre_process
        ):
            continue
        for weight_name, sharded_info in _prepare_metadata(prefix, submodule).items():
            infos[f"{prefix}.{weight_name}"] = sharded_info
    return infos
