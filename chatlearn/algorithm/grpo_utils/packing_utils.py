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
"""Packing Utilities"""

from typing import List, Dict, Union, Any
import warnings

import torch
import numpy as np
from sortedcontainers import SortedList

def bin_packing(seq_len_list: List[int], max_train_token: int):
    """
    Implementation of best fit decreasing bin packing algorithm
    """
    if sum(np.array(seq_len_list) > max_train_token) >  0:
        message = f"Max length in microbatch exceeds max_token_in_seq, max length: {max(seq_len_list)}, max_train_token: {max_train_token}"
        warnings.warn(message)

    seqlen_id_mapping = dict(enumerate(seq_len_list))
    sorted_mapping = dict(sorted(seqlen_id_mapping.items(), key=lambda item: item[1], reverse=True))
    bins_id = []
    bins_seqlen = []
    for key, value in sorted_mapping.items():
        min_space_left = None
        best_bin_index = None
        for id_, bin_ in enumerate(bins_seqlen):
            space_left = max_train_token - sum(bin_)
            if space_left > value:
                if min_space_left is None:
                    min_space_left = space_left
                    best_bin_index = id_
                else:
                    if space_left < min_space_left:
                        min_space_left = space_left
                        best_bin_index = id_
        if best_bin_index is None:
            bins_id.append([key])
            bins_seqlen.append([value])
        else:
            bins_id[best_bin_index].append(key)
            bins_seqlen[best_bin_index].append(value)
    return list(bins_id), list(bins_seqlen)

def adjust_bins(bins_id, bins_num):
    """
    Adjust bins to balance total sequence length.
    First create a sorted list of (total_seq_len, -min_seq_len, sorted_bin_list).
    This will make sure last element in sorted list is bin will largest total_seq_len 
    and smallest single sample sequence length.
    For each round, pop the smallest bin from tail and add the smallest sample to head.
    The loop will stop if moving sample will no longer increase balance.
    """
    if len(bins_id) == 1:
        return bins_id, bins_num
    sorted_list = SortedList()
    for i in range(len(bins_id)):
        min_seq = bins_num[i][-1] if len(bins_num[i]) > 0 else 0
        sorted_list.add((
            sum(bins_num[i]),
            -min_seq,
            SortedList([(num, id_) for id_, num in zip(bins_id[i], bins_num[i])])))
    # Balance sorted_list
    stop = False
    while not stop:
        min_sum, _, min_bin = sorted_list.pop(0)
        max_sum, _, max_bin = sorted_list.pop(-1)
        smallest_num, smallest_id = max_bin.pop(0)
        if abs((max_sum - min_sum - 2 * smallest_num)) < max_sum - min_sum:
            min_bin.add((smallest_num, smallest_id))
            sorted_list.add((max_sum - smallest_num, -max_bin[0][0], max_bin))
            sorted_list.add((min_sum + smallest_num, -min_bin[0][0], min_bin))
        else:
            stop = True
            max_bin.add((smallest_num, smallest_id))
            sorted_list.add((max_sum, -max_bin[0][0], max_bin))
            sorted_list.add((min_sum, -min_bin[0][0], min_bin))
    bins_id = [[item[1] for item in list_[2]] for list_ in sorted_list]
    bins_seq = [[item[0] for item in list_[2]] for list_ in sorted_list]
    bins_id.reverse()
    bins_seq.reverse()
    return bins_id, bins_seq

def prepare_packing_attn_mask(total_seq_len_list: List[int], pad_size: int, dtype):
    total_seq_length = sum(total_seq_len_list) + pad_size
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((total_seq_length, total_seq_length), fill_value=min_dtype, dtype=dtype)
    seqlen_gather = 0
    for seq_now in total_seq_len_list:
        seq_now = total_seq_len_list[i]
        tri_mask = torch.tril(torch.ones((seq_now, seq_now), dtype=dtype)) == 0
        causal_mask[seqlen_gather : seqlen_gather + seq_now, seqlen_gather : seqlen_gather + seq_now] *= tri_mask
        seqlen_gather += seq_now
    causal_mask = causal_mask[None, None, :, :].expand(1, 1, -1, -1)
    return causal_mask

def regroup_data_from_list(data_all: Dict, data_position: List[int]):
    if isinstance(data_all, torch.Tensor):
        return data_all[torch.tensor(data_position)]
    if isinstance(data_all, list):
        return [data_all[item] for item in data_position]


def regroup_data_packing(
    data_list: List[Dict[str, Union[torch.Tensor, List[Any]]]],
    process_group_list: List[Any],
    max_train_token: int,
    offset: int
) -> List[Dict[str, Union[torch.Tensor, List[Any]]]]:
    """Automatically split a list of data into serveral micro-batches according to
    `max_train_token`, used for dynamic-batching. Note that the data in each batch
    is still not merged into one packed sample.

    Args:
        data_list (List[Dict[str, Union[torch.Tensor, List[Any]]]]): A list of 
        PRE-BATCHED data to be regrouped.
        max_train_token (int): The maximum token num in each batch.
    
    Returns:
        a list of micro-batches. Each micro-batch is a list of samples without batching.
    """
    # data_b should contain all data in one microbatch
    regroup_data_list = []
    # Get train tokens for whole minibatch
    total_token_length = [
        data_b["prompt_token_length"] + data_b["response_token_length"]
        for data_b in data_list
    ]
    # Get bin_packing result
    bins_id, bins_seq = bin_packing(seq_len_list=total_token_length, max_train_token=max_train_token)
    local_bin_size = len(bins_id)
    bin_size = torch.tensor(local_bin_size).cuda()
    # Get max_bin_size across all rank in same model replica
    # For megatron, all_reduce along mp group first and emp group second
    # For FSDP, all_reduce along default group
    if process_group_list is None:
        process_group_list = [None]
    for pg in process_group_list:
        torch.distributed.all_reduce(bin_size, op=torch.distributed.ReduceOp.MAX, group=pg)
    max_bin_size = bin_size.cpu().item()
    for _ in range(max_bin_size - local_bin_size):
        bins_id.append([])
        bins_seq.append([])
    bins_id, bins_seq = adjust_bins(bins_id, bins_seq)
    # Prepare train data for each micro batch
    for micro_batch_id in bins_id:
        regroup_data_list.append([])
        for sample_id in micro_batch_id:
            packed_sample = data_list[sample_id]
            packed_sample.update({'id_in_list': sample_id + offset})
            regroup_data_list[-1].append(packed_sample)
    return regroup_data_list
