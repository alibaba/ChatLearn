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
"""helper to collect shape infos for vLLM model"""
from typing import Dict
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding
)
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)

from .sharded_tensor_info import ShardedTensorInfo

def _prepare_metadata(module: nn.Module):
    results = {}
    if isinstance(module, RMSNorm):
        results['weight'] = ShardedTensorInfo.from_global_shape(
            tuple(module.weight.shape), dtype=module.weight.dtype
        )
    elif isinstance(module, (
        ColumnParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
        ParallelLMHead,
        VocabParallelEmbedding
    )):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        w, h = module.weight.shape
        if isinstance(module, RowParallelLinear):
            global_shape=(w, h * tp_size)
            axis_fragmentations=(1, tp_size)
            global_offset=(0, tp_rank)
        elif isinstance(module, (
            ColumnParallelLinear, ParallelLMHead, ParallelLMHead, VocabParallelEmbedding)):
            axis_fragmentations=(tp_size, 1)
            global_offset=(tp_rank, 0)
            global_shape=(w * tp_size, h)
        else:
            axis_fragmentations=(1, 1)
            global_offset=(0, 0)
            global_shape=(w, h)
        results['weight'] = ShardedTensorInfo(
            dtype=module.weight.dtype,
            global_shape=global_shape,
            axis_fragmentations=axis_fragmentations,
            global_offset=global_offset
        )
        bias = getattr(module, 'bias', None)
        if bias is not None:
            results['bias'] = ShardedTensorInfo(
                dtype=bias.dtype,
                global_shape=global_shape[:1],
                axis_fragmentations=axis_fragmentations[:1],
                global_offset=global_offset[:1]
            )
        e_score_correction_bias = getattr(module, 'e_score_correction_bias', None)
        if e_score_correction_bias is not None:
            results['e_score_correction_bias'] = ShardedTensorInfo.from_global_shape(
                dtype=e_score_correction_bias.dtype,
                global_shape=e_score_correction_bias.shape
            )
    elif isinstance(module, FusedMoE):
        if hasattr(module, 'moe_parallel_config'):
            parallel_config = module.moe_parallel_config
            ep_rank, ep_size = module.ep_rank, module.ep_size
            tp_rank, tp_size = module.tp_rank, module.tp_size
            use_ep = parallel_config.use_ep
        else:
            ep_rank, ep_size = module.ep_rank, module.ep_size
            tp_rank, tp_size = module.tp_rank, module.tp_size
            dp_size = module.dp_size
            use_ep = (
                get_current_vllm_config().parallel_config.enable_expert_parallel and
                tp_size * dp_size > 1
            )
        # pylint: disable=unnecessary-lambda-assignment
        smallest_multiple = lambda x, k: x + ((k - x % k) % k)
        if use_ep:
            # NOTE: currently vllm ep is not supported in parameter sync.
            padded_global_num_experts = smallest_multiple(module.global_num_experts, ep_size)
            _, w, h = module.w13_weight.shape
            results['w13_weight'] = ShardedTensorInfo(
                dtype=module.w13_weight.dtype,
                local_shape=module.w13_weight.shape,
                global_shape=(padded_global_num_experts, w, h),
                axis_fragmentations=(ep_size, 1, 1),
                global_offset=(ep_rank, 0, 0),
            )
            results['w2_weight'] = ShardedTensorInfo(
                dtype=module.w13_weight.dtype,
                local_shape=module.w13_weight.shape,
                global_shape=(padded_global_num_experts, w, h),
                axis_fragmentations=(ep_size, 1, 1),
                global_offset=(ep_rank, 0, 0),
            )
        else:
            l, w, h = module.w13_weight.shape
            results['w13_weight'] = ShardedTensorInfo(
                dtype=module.w13_weight.dtype,
                global_shape=(l, w * tp_size, h),
                axis_fragmentations=(1, tp_size, 1),
                global_offset=(0, tp_rank, 0),
            )
            l, w, h = module.w2_weight.shape
            results['w2_weight'] = ShardedTensorInfo(
                dtype=module.w2_weight.dtype,
                global_shape=(l, w, h * tp_size),
                axis_fragmentations=(1, 1, tp_size),
                global_offset=(0, 0, tp_rank),
            )
    return results

def build_sharded_info_for_vllm_model(
    model: nn.Module
) -> Dict[str, ShardedTensorInfo]:
    """build sharded tensor info from vLLM Model

    Args:
        model (vLLMModel): The given vLLM model

    Returns:
        Dict[str, ShardedTensorInfo]: A dict maps local parameter
        name to sharded_info
    """
    infos = {}
    for prefix, submodule in model.named_modules():
        for weight_name, sharded_info in _prepare_metadata(submodule).items():
            infos[f"{prefix}.{weight_name}"] = sharded_info
    return infos
