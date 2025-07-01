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
from copy import deepcopy
from typing import *
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

def process_gate_up_tensor(
    sharded_info: ShardedTensorInfo,
    dst_tp_size: int
) -> List[Tuple[ShardedTensorInfo, ...]]:
    """The weight/bias of gate_up_proj is represent

    Args:
        sharded_info (ShardedTensorInfo): _description_
        dst_tp_size (int): _description_

    Yields:
        Generator[Tuple[int, ShardedTensorInfo, ShardedTensorInfo]]: _description_
    """
    src_tp_rank = sharded_info.global_offset[0]
    src_tp_size = sharded_info.axis_fragmentations[0]
    intermediates = ShardedTensorInfo.from_global_shape(
        sharded_info.global_shape
    ).fragment(src_tp_size * 2)

    # TODO: Here we may generate more fragments than expected, which could 
    # TODO: reduce the comm efficiency, fix later
    gate_mappings, up_mappings = [], []
    for gate_part in intermediates[src_tp_rank].fragment(dst_tp_size * 2):
        dst_tp_rank = gate_part.global_offset[0]
        gate_mappings.append((
            (
                gate_part
                .refragment(src_tp_size * 2)
                .map_to_frag_id(src_tp_rank * 2)
                .refragment(src_tp_size)
            ),
            (
                gate_part
                .map_to_frag_id(dst_tp_rank * 2)
                .refragment(dst_tp_size)
            ),            
        ))
    for up_part in intermediates[src_tp_size + src_tp_rank].fragment(dst_tp_size * 2):
        dst_tp_rank = up_part.global_offset[0] - dst_tp_size
        up_mappings.append((
            (
                up_part
                .refragment(src_tp_size * 2)
                .map_to_frag_id(src_tp_rank * 2 + 1)
                .refragment(src_tp_size)
            ),
            (
                up_part
                .map_to_frag_id(dst_tp_rank * 2 + 1)
                .refragment(dst_tp_size)
            ),            
        ))
    return gate_mappings + up_mappings

def _build_frag_mapping(
    num_heads: int,
    num_query_group: int, 
    src_tp_size: int, 
    dst_tp_size: int
):
    """Generate a mapping between mcore qkv heads
    and vllm qkv heads.

    Mcore layout of first dim per tp rank when 
    nh=24, ng=8, tp=4, nq=3: [q q q k v q q q k v], 
    while vLLM: [q q q q q q k k v v]

    Args:
        num_heads (int): _description_
        num_query_group (int): _description_
        src_tp_size (int): _description_
        dst_tp_size (int): _description_
    """
    flatten = lambda x: list(chain.from_iterable(x))
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
            [f"k{g_id}" for g_id in range(num_query_group // dst_tp_size)] +
            [f"v{g_id}" for g_id in range(num_query_group // dst_tp_size)]
            for r_id in range(dst_tp_size)
        ])
    else:
        vllm_layout = flatten([
            [f"q{r_id * vllm_nq + q_id}" for q_id in range(num_heads // dst_tp_size)] + 
            [f"k{r_id // num_query_group}", f"v{r_id // num_query_group}"]
            for r_id in range(dst_tp_size)
        ])
    return {
        x: mcore_layout.index(n) for x, n in enumerate(vllm_layout)
    }

def process_qkv_tensor(
    sharded_info: ShardedTensorInfo,
    num_heads: int,
    num_query_groups: Optional[int],
    dst_tp_size: int
) -> List[Tuple[ShardedTensorInfo, ...]]:
    """Process qkv weight/bias to generate shard mapping.

    Args:
        sharded_info (ShardedTensorInfo): _description_
        num_heads (int): _description_
        num_query_group (int): _description_
        dst_tp_size (int): _description_

    """
    if num_query_groups is None:
        num_query_groups = num_heads
    if num_query_groups % dst_tp_size != 0 and dst_tp_size % num_query_groups != 0:
        raise ValueError(f"num_query_groups {num_query_groups} must be divisible or multiple by dst_tp_size {dst_tp_size}")
    src_tp_size = sharded_info.axis_fragmentations[0]
    src_global_shape = sharded_info.global_shape
    vllm_id_to_mcore_id = _build_frag_mapping(
        num_heads,
        num_query_groups,
        src_tp_size,
        dst_tp_size
    )

    mcore_id_to_frags = {
        part.global_offset[0]: part.refragment(src_tp_size)
        for part in sharded_info.fragment(num_query_groups * (2 + num_heads // num_query_groups))
    }

    head_dim = sharded_info.global_shape[0] // (num_heads + 2 * num_query_groups)
    total_heads = num_heads + 2 * num_query_groups * max(1, dst_tp_size // num_query_groups)

    results = []
    for vllm_idx, dst_part in enumerate(ShardedTensorInfo.from_global_shape(
        (total_heads * head_dim, ) + src_global_shape[1:]
    ).fragment(total_heads)):
        mcore_idx = vllm_id_to_mcore_id[vllm_idx]
        if mcore_idx not in mcore_id_to_frags:
            continue
        results.append((
            mcore_id_to_frags[mcore_idx],
            dst_part.refragment(dst_tp_size)
        ))
    return results