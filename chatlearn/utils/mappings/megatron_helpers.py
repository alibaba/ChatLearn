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
import torch

from typing import TYPE_CHECKING, List, Dict
from torch import nn

from .sharded_tensor_info import ShardedTensorInfo

if TYPE_CHECKING:
    from megatron.core.models.gpt import GPTModel

try:
    from megatron.core import mpu
    from megatron.core.extensions.transformer_engine import (
        TEGroupedLinear,
        TELinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
    )
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.tensor_parallel import (
        VocabParallelEmbedding,
        ColumnParallelLinear
    )
    HAVE_MEGATRON = True
except:
    HAVE_MEGATRON = False


def _prepare_metadata(module: nn.Module):
    if not HAVE_MEGATRON:
        raise SystemError("Cannot call this function without megatron")
    results = {}
    if isinstance(module, TENorm):
        results['weight'] = ShardedTensorInfo.from_global_shape(
            tuple(module.weight.shape), dtype=module.weight.dtype
        )
        if hasattr(module, 'bias'):
            results['bias'] = ShardedTensorInfo.from_global_shape(
                tuple(module.bias.shape), dtype=module.bias.dtype
            )
    elif isinstance(module, (TELinear, TELayerNormColumnParallelLinear, ColumnParallelLinear)):
        if module.is_expert:
            tp_rank = mpu.get_expert_tensor_parallel_rank()
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()
        parallel_mode = 'column' if isinstance(module, ColumnParallelLinear) else module.parallel_mode
        if parallel_mode == 'row':
            global_shape=(w, h * tp_size)
            axis_fragmentations=(1, tp_size)
            global_offset=(0, tp_rank)
        elif parallel_mode == 'column':
            axis_fragmentations=(tp_size, 1)
            global_offset=(tp_rank, 0)
            global_shape=(w * tp_size, h)
        else:
            axis_fragmentations=(1, 1)
            global_offset=(0, 0)
            global_shape=(w, h)  
        w, h = module.weight.shape
        results['weight'] = ShardedTensorInfo(
            dtype=module.weight.dtype,
            global_shape=global_shape,
            axis_fragmentations=axis_fragmentations,
            global_offset=global_offset
        )
        if module.bias is not None:
            results['bias'] = ShardedTensorInfo(
                dtype=module.bias.dtype,
                global_shape=global_shape[:1],
                axis_fragmentations=axis_fragmentations[:1],
                global_offset=global_offset[:1]
            )
        
        if hasattr(module, 'layer_norm_weight'):
            results['layer_norm_weight'] = ShardedTensorInfo.from_global_shape(
                module.layer_norm_weight.shape,
                dtype=module.layer_norm_weight.dtype
            )
        if hasattr(module, 'layer_norm_bias'):
            results['layer_norm_bias'] = ShardedTensorInfo.from_global_shape(
                module.layer_norm_bias.shape,
                dtype=module.layer_norm_bias.dtype
            )
    elif isinstance(module, TEGroupedLinear):
        if module.is_expert:
            tp_rank = mpu.get_expert_tensor_parallel_rank()
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()

        for gemm_id in range(module.num_gemms):
            weight = getattr(module, f"weight{gemm_id}")
            w, h = weight.shape
            if module.parallel_mode == 'row':
                global_shape=(w, h * tp_size)
                axis_fragmentations=(1, tp_size)
                global_offset=(0, tp_rank)
            elif module.parallel_mode == 'column':
                axis_fragmentations=(tp_size, 1)
                global_offset=(tp_rank, 0)
                global_shape=(w * tp_size, h)
            else:
                raise NotImplementedError("duplicated linear is not supported")
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
        tp_rank = module.tp_group.rank()
        tp_size = module.tp_group.size()
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
    """build sharded tensor info from GPTModel

    Args:
        model (GPTModel): The given model

    Returns:
        Dict[str, ShardedTensorInfo]: A dict maps local parameter 
        name to sharded_info
    """
    # TODO: can we parse sharded info from sharded_state_dict?
    infos = {}
    for prefix, submodule in model.named_modules():
        for weight_name, sharded_info in _prepare_metadata(submodule).items():
            infos[f"{prefix}.{weight_name}"] = sharded_info
    return infos