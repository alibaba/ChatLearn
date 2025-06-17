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
"""Trainer Utilities"""
from typing import List
import warnings

import torch
import torch.nn.functional as F

import numpy as np

def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Generate logprobs from logits
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

def sp_split(input_tensor, split_dim, sp_size, sp_local_rank):
    """
    split input by sp_size
    """
    return torch.chunk(input_tensor, sp_size, split_dim)[sp_local_rank]

def generate_loss_mask_position_ids(tokens: torch.Tensor, prompt_token_length: list, response_token_length:list):
    """
    Setup loss_mask and position_ids by prompt token length and response token length
    """
    # For post-training, we only train on response tokens
    loss_mask = torch.zeros_like(tokens, dtype=torch.int32, device=tokens.device)
    attn_mask = torch.zeros_like(tokens, dtype=torch.int32, device=tokens.device)
    for i in range(len(prompt_token_length)): # pylint: disable=consider-using-enumerate
        loss_mask[i, prompt_token_length[i]: prompt_token_length[i] + response_token_length[i]] = 1.0
        attn_mask[i, : prompt_token_length[i] + response_token_length[i]] = 1.0
    _, seq_len = tokens.size()
    position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0).expand_as(tokens)

    return attn_mask, loss_mask, position_ids

def bin_packing(seq_len_list, max_train_token):
    """
    Implementation of best fit decreasing bin packing algorithm
    """
    if sum(np.array(seq_len_list) > max_train_token) >  0:
        message = f"Max length in microbatch exceeds max_token_in_seq, max length: {max(seq_len_list)}, max_train_token: {max_train_token}"
        warnings.warn(message)

    seqlen_id_mapping = {i: seq_len for i, seq_len in enumerate(seq_len_list)}
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

def bin_packing_fix_bin(seq_len_list, bin_size):
    """
    Implementation of best fit decreasing bin packing algorithm with fix bin size
    """
    seqlen_id_mapping = {i: seq_len for i, seq_len in enumerate(seq_len_list)}
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

def prepare_packing_attn_mask(total_seq_len_list, dtype, pad_size):
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

def regroup_data_from_list(data_all, data_position):
    if isinstance(data_all, torch.Tensor):
        return data_all[torch.tensor(data_position)]
    if isinstance(data_all, list):
        return [data_all[item] for item in data_position]

def regroup_data_packing(data_list, key_list, max_train_token):
    # data_b should contain all data in one microbatch
    data_b = data_list[0]
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
    # Prepare train data for each micro batch
    for micro_batch_id, micro_batch_seqlen in zip(bins_id, bin_seqlen):
        data_new = {}
        for key in key_list:
            data_new[key] = regroup_data_from_list(data_b[key], micro_batch_id)
        data_new["bin_ids"] = micro_batch_id
        data_new["bin_seqlen"] = micro_batch_seqlen
        regroup_data_list.append(data_new)
    return regroup_data_list
