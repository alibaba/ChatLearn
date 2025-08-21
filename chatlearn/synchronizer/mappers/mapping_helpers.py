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
"""helper function to map layout between MCore and vLLM"""
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
from itertools import chain

from chatlearn.utils.mappings import ShardedTensorInfo

def process_normal_tensor(
    sharded_info: ShardedTensorInfo,
    dst_tp_size: int,
    axis: int=0
) -> List[Tuple[ShardedTensorInfo, ...]]:
    """For normal parameters which have the consistent layout,
    only resharding is required.

    Example: Assume a normal ColumnLinear with shape [16, 8] in MCore,
    When TP=2, each rank will get ShardedTensorInfo(local_shape=[8, 8],
    global_shape=[16, 8], global_offset=...). If we want to copy part
    of weight to the vLLM ColumnLinear of TP=4, reshard with TP=4 will
    further split weight of this rank and then we can apply the copy
    with the returned sharded_info.

    Args:
        sharded_info (ShardedTensorInfo): The sharded_info of the input tensor
        dst_tp_size (int): The target tp_size
        axis (int, optional): The axis to be resharded.

    """
    return [
        (
            tensor_part_info.refragment(sharded_info.axis_fragmentations[axis], axis),
            tensor_part_info
        ) for tensor_part_info in sharded_info.fragment(dst_tp_size, axis)
    ]

def _build_gate_up_layout(src_tp_size: int, dst_tp_size: int):
    """
    build layout with 2 * lcm(src_tp, dst_tp) chunks
    """
    n_chunks = math.lcm(src_tp_size, dst_tp_size)
    flatten = lambda x: list(chain.from_iterable(x)) # pylint: disable=unnecessary-lambda-assignment
    mcore_layout = flatten([
        [ f"g{c_id + tp_rank * (n_chunks // src_tp_size)}" for c_id in range(n_chunks // src_tp_size) ] +
        [ f"u{c_id + tp_rank * (n_chunks // src_tp_size)}" for c_id in range(n_chunks // src_tp_size) ]
        for tp_rank in range(src_tp_size)
    ])

    vllm_layout = flatten([
        [ f"g{c_id + tp_rank * (n_chunks // dst_tp_size)}" for c_id in range(n_chunks // dst_tp_size) ] +
        [ f"u{c_id + tp_rank * (n_chunks // dst_tp_size)}" for c_id in range(n_chunks // dst_tp_size) ]
        for tp_rank in range(dst_tp_size)
    ])

    return mcore_layout, vllm_layout

def process_gate_up_tensor(
    sharded_info: ShardedTensorInfo,
    dst_tp_size: int,
    proj_type: Literal['gate_up_proj', 'gate_proj', 'up_proj']
) -> List[Tuple[ShardedTensorInfo, ...]]:
    """The weight/bias of gate_up_proj is represent

    Args:
        sharded_info (ShardedTensorInfo): The ShardedTensorInfo of the
        input tensor from gate up proj layer
        dst_tp_size (int): the vLLM tp_size

    Returns:
        List[Tuple[ShardedTensorInfo, ...]]: The layout mapping.
    """
    src_tp_size = sharded_info.axis_fragmentations[0]

    mcore_layout, vllm_layout = _build_gate_up_layout(src_tp_size, dst_tp_size)
    mcore_id_to_frags = {
        part.global_offset[0]: part.refragment(src_tp_size)
        for part in sharded_info.fragment(math.lcm(src_tp_size, dst_tp_size) * 2)
    }
    if proj_type == 'gate_up_proj':
        n_chunks = math.lcm(src_tp_size, dst_tp_size) * 2
        full_dst_info = ShardedTensorInfo.from_global_shape(sharded_info.global_shape)
    else:
        n_chunks = math.lcm(src_tp_size, dst_tp_size)
        full_dst_info = ShardedTensorInfo.from_global_shape(
            (sharded_info.global_shape[0] // 2, ) + sharded_info.global_shape[1:]
        )

    results = []
    for chunk_idx, dst_part in enumerate(full_dst_info.fragment(n_chunks)):
        if proj_type == 'gate_up_proj':
            chunk_name = vllm_layout[chunk_idx]
        else:
            chunk_name = f"{proj_type[:1]}{chunk_idx}"
        mcore_idx = mcore_layout.index(chunk_name)
        if mcore_idx not in mcore_id_to_frags:
            continue
        results.append((
            mcore_id_to_frags[mcore_idx],
            dst_part.refragment(dst_tp_size)
        ))
    return __maybe_merge(results)


def _build_qkv_layout(
    num_heads: int,
    num_query_group: int,
    dst_tp_size: int
):
    """Generate a mapping between mcore qkv heads (mix-style qkv)
    and vllm qkv heads (no mix-style qkv).

    Mcore layout of first dim per tp rank when
    nh=24, ng=8, tp=4, nq=3: [q q q k v q q q k v],
    while vLLM: [q q q q q q k k v v]

    Args:
        num_heads (int): The num of attention heads
        num_query_group (int): The num of query groups
        dst_tp_size (int): The dst tensor parallel size
    """
    flatten = lambda x: list(chain.from_iterable(x)) # pylint: disable=unnecessary-lambda-assignment
    nq = num_heads // num_query_group
    mcore_layout = []
    vllm_layout = []
    mcore_layout = flatten([
        [ f"q{g_id * nq + q_id}" for q_id in range(nq)] + [f"k{g_id}", f"v{g_id}"]
        for g_id in range(num_query_group)
    ])
    vllm_nq = num_heads // dst_tp_size
    if dst_tp_size < num_query_group:
        vllm_layout = flatten([
            [f"q{r_id * vllm_nq + q_id}" for q_id in range(num_heads // dst_tp_size)] +
            [f"k{g_id + r_id * (num_query_group // dst_tp_size)}" for g_id in range(num_query_group // dst_tp_size)] +
            [f"v{g_id + r_id * (num_query_group // dst_tp_size)}" for g_id in range(num_query_group // dst_tp_size)]
            for r_id in range(dst_tp_size)
        ])
    else:
        vllm_layout = flatten([
            [f"q{r_id * vllm_nq + q_id}" for q_id in range(num_heads // dst_tp_size)] +
            [f"k{r_id * num_query_group // dst_tp_size}", f"v{r_id * num_query_group // dst_tp_size}"]
            for r_id in range(dst_tp_size)
        ])
    return mcore_layout, vllm_layout


def process_qkv_tensor(
    sharded_info: ShardedTensorInfo,
    num_heads: int,
    num_query_groups: Optional[int],
    dst_tp_size: int,
    proj_type: Literal['qkv_proj', 'q_proj', 'k_proj', 'v_proj']
) -> List[Tuple[ShardedTensorInfo, ...]]:
    """Process qkv weight/bias to generate shard mapping.

    Args:
        sharded_info (ShardedTensorInfo): The sharded info representing megatron mixed qkv
        num_heads (int): The number of attention heads
        num_query_group (int): The number of query groups
        dst_tp_size (int): The target tensor parallel size
        proj_type (Literal['qkv_proj', 'q_proj', 'k_proj', 'v_proj']): the projection type
    """
    if num_query_groups is None:
        num_query_groups = num_heads
    if num_query_groups % dst_tp_size != 0 and dst_tp_size % num_query_groups != 0:
        raise ValueError(f"num_query_groups {num_query_groups} must be divisible or multiple by dst_tp_size {dst_tp_size}")
    head_dim = sharded_info.global_shape[0] // (num_heads + 2 * num_query_groups)
    src_tp_size = sharded_info.axis_fragmentations[0]
    src_global_shape = sharded_info.global_shape

    mcore_layout, vllm_layout = _build_qkv_layout(num_heads, num_query_groups, dst_tp_size)
    mcore_id_to_frags = {
        part.global_offset[0]: part.refragment(src_tp_size)
        for part in sharded_info.fragment(num_query_groups * (2 + num_heads // num_query_groups))
    }

    if proj_type == 'qkv_proj':
        n_heads = num_heads + 2 * num_query_groups * max(1, dst_tp_size // num_query_groups)
    elif proj_type == 'q_proj':
        n_heads = num_heads
    else:
        n_heads = num_query_groups * max(1, dst_tp_size // num_query_groups)
    full_dst_info = ShardedTensorInfo.from_global_shape((n_heads * head_dim, ) + src_global_shape[1:])

    results = []
    for head_idx, dst_part in enumerate(full_dst_info.fragment(n_heads)):
        if proj_type == 'qkv_proj':
            head_name = vllm_layout[head_idx]
        else:
            head_name = f"{proj_type[:1]}{head_idx}" # q0 / k1 / v2, etc.
        mcore_idx = mcore_layout.index(head_name)
        if mcore_idx not in mcore_id_to_frags:
            continue
        results.append((
            mcore_id_to_frags[mcore_idx],
            dst_part.refragment(dst_tp_size)
        ))
    return __maybe_merge(results)

def __maybe_merge(mappings: List[Tuple[ShardedTensorInfo, ShardedTensorInfo]], axis: int=0):
    """Try to merge adjacent shards to reduce the number of shards."""

    mappings = sorted(mappings, key=lambda x: (x[0].global_offset[axis], x[0].local_offset[axis]))
    results = []
    to_be_merged = []
    for src_part, dst_part in mappings:
        if len(to_be_merged) == 0:
            to_be_merged.append((src_part, dst_part))
            continue

        if (
            to_be_merged[-1][0].local_offset[axis] + to_be_merged[-1][0].local_shape[axis] == src_part.local_offset[axis] and
            to_be_merged[-1][1].local_offset[axis] + to_be_merged[-1][1].local_shape[axis] == dst_part.local_offset[axis] and
            to_be_merged[-1][0].global_offset[axis] == src_part.global_offset[axis] and
            to_be_merged[-1][1].global_offset[axis] == dst_part.global_offset[axis]
        ):
            to_be_merged.append((src_part, dst_part))
        else:
            results.append((
                ShardedTensorInfo.concat([item[0] for item in to_be_merged], axis),
                ShardedTensorInfo.concat([item[1] for item in to_be_merged], axis)
            ))
            to_be_merged = [(src_part, dst_part)]

    results.append((
        ShardedTensorInfo.concat([item[0] for item in to_be_merged], axis),
        ShardedTensorInfo.concat([item[1] for item in to_be_merged], axis)
    ))
    return results

@dataclass(frozen=True)
class VLLM_HELPERS:
    """The mapper configs for vllm"""
    merge_gate_up = True
    merge_qkv = True
    merge_expert = True
    force_full_model = False


@dataclass(frozen=True)
class HF_HELPERS:
    """The mapper configs for huggingface/sglang"""
    merge_gate_up = False
    merge_qkv = False
    merge_expert = False
    force_full_model = True
