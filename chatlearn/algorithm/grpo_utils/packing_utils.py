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

from typing import List, Dict
import warnings

import torch

import numpy as np
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

def bin_packing_fix_bin(seq_len_list: List[int], bin_size: int):
    """
    Implementation of best fit decreasing bin packing algorithm with fix bin size
    """
    seqlen_id_mapping = dict(enumerate(seq_len_list))
    sorted_mapping = dict(sorted(seqlen_id_mapping.items(), key=lambda item: item[1], reverse=True))
    bins_id = [[] for i in range(bin_size)]
    bins_seqlen = [[] for i in range(bin_size)]
    for key, value in sorted_mapping.items():
        min_sum = None
        for id_, bin_ in enumerate(bins_seqlen):
            bin_sum = value + sum(bin_)
            if min_sum is None:
                min_sum = bin_sum
                best_bin_index = id_
            else:
                if bin_sum < min_sum:
                    min_sum = bin_sum
                    best_bin_index = id_
        bins_id[best_bin_index].append(key)
        bins_seqlen[best_bin_index].append(value)
    # sort bins by seqlen in  single bin
    bins_seqlen_sum = [sum(bin_seqlen) for bin_seqlen in bins_seqlen]
    sorted_bin = sorted(zip(bins_seqlen_sum, bins_id))
    sorted_binseq = sorted(zip(bins_seqlen_sum, bins_seqlen))
    _, bins_id = zip(*sorted_bin)
    _, bins_seqlen = zip(*sorted_binseq)
    return list(bins_id), list(bins_seqlen)

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

def merge_data_list(data_list: List):
    if len(data_list) == 1:
        return data_list[0]
    # Extract all data
    merged_data = {}
    for key in data_list[0]:
        merged_data[key] = []
    for data_all in data_list:
        for key in data_all:
            merged_data[key].append(data_all[key])
    for key in merged_data: # pylint: disable=consider-using-dict-items
        if isinstance(merged_data[key][0], torch.Tensor):
            merged_data[key] = torch.cat(merged_data[key], dim=0)
        elif isinstance(merged_data[key][0], list):
            merged_data[key] = [item for sub_list in merged_data[key] for item in sub_list]
    return merged_data

def regroup_data_packing(data_list: List, key_list: List[str], max_train_token: int):
    # data_b should contain all data in one microbatch
    data_b = merge_data_list(data_list)
    regroup_data_list = []
    # Get train tokens for whole minibatch
    total_token_length = [
        prompt_len + response_len
        for prompt_len, response_len
        in zip(data_b["prompt_token_length"], data_b["response_token_length"])
    ]
    # Get bin_packing result
    bins_id, bin_seqlen = bin_packing(seq_len_list=total_token_length, max_train_token=max_train_token)
    bin_size = torch.tensor(len(bins_id)).cuda()
    # Get max_bin_size across all dp rank
    torch.distributed.all_reduce(bin_size, op=torch.distributed.ReduceOp.MAX)
    bins_id, bin_seqlen = bin_packing_fix_bin(seq_len_list=total_token_length, bin_size=bin_size.cpu().item())
   # print(f"debugyy bin_size: {len(bins_id)}, sample in each bins: {[sum(bins) for bins in bin_seqlen]}")
    # Prepare train data for each micro batch
    for micro_batch_id, micro_batch_seqlen in zip(bins_id, bin_seqlen):
        data_new = {}
        for key in key_list:
            data_new[key] = regroup_data_from_list(data_b[key], micro_batch_id)
        data_new["bin_ids"] = micro_batch_id
        data_new["bin_seqlen"] = micro_batch_seqlen
        regroup_data_list.append(data_new)
    return regroup_data_list
