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
from collections import defaultdict
from typing import List, Any, Dict, Optional
import math

import torch
import torch.nn.functional as F
from flash_attn.bert_padding import index_first_axis
from einops import rearrange
from chatlearn.algorithm.grpo_utils.packing_utils import regroup_data_packing


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy

def entropy_from_logits_with_chunking(logits: torch.Tensor, chunk_size: int = 512):
    """Memory-efficient entropy calculation with chunking.
    logits shape: (batch_size, seq_len, dimension)
    entropy shape: (batch_size, seq_len)
    """
    entropy = torch.zeros(logits.shape[:-1], device=logits.device)
    for i in range(0, logits.shape[1], chunk_size):
        logits_chunk = logits[: ,i : i + chunk_size]
        pd_chunk = torch.nn.functional.softmax(logits_chunk, dim=-1)
        entropy_chunk = torch.logsumexp(logits_chunk, dim=-1) - torch.sum(pd_chunk * logits_chunk, dim=-1)
        entropy[:, i : i + chunk_size] = entropy_chunk
    return entropy

def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Generate logprobs from logits
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

def sp_split(input_tensor:torch.Tensor, split_dim:int, sp_size:int, sp_local_rank:int):
    """
    split input by sp_size
    """
    return torch.chunk(input_tensor, sp_size, split_dim)[sp_local_rank]

def generate_loss_mask_position_ids(
    tokens: torch.Tensor,
    prompt_token_length: List[int],
    response_token_length: List[int],
    prompt_position_ids: Optional[List[List[int]]] = None
):
    """
    Setup loss_mask and position_ids by prompt token length and response token length
    """
    # For post-training, we only train on response tokens
    loss_mask = torch.zeros_like(tokens, dtype=torch.int32, device=tokens.device)
    attn_mask = torch.zeros_like(tokens, dtype=torch.int32, device=tokens.device)
    for i in range(len(prompt_token_length)): # pylint: disable=consider-using-enumerate
        loss_mask[i, prompt_token_length[i]: prompt_token_length[i] + response_token_length[i]] = 1.0
        attn_mask[i, : prompt_token_length[i] + response_token_length[i]] = 1.0
    batch_size, seq_len = tokens.size()

    if prompt_position_ids:
        # vl preprocess
        position_ids = torch.zeros((3, batch_size, seq_len), dtype=torch.long, device=tokens.device)
        for batch_idx, batch in enumerate(prompt_position_ids):
            for list_idx in range(3):
                sublist = batch[list_idx]
                # Fill the beginning of the result tensor with the existing sublist
                position_ids[list_idx, batch_idx, :len(sublist)] = torch.tensor(sublist)
                # Determine the starting value for padding
                start_value = max(sublist) + 1
                # Fill the rest with increasing values starting from max(sublist) + 1
                position_ids[list_idx, batch_idx, len(sublist):] = torch.arange(
                    start_value, start_value + (seq_len - len(sublist))
                )
    else:
        # text only
        position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)

    return attn_mask, loss_mask, position_ids

def split_microbatch(
    data_list: List,
    micro_batch_size: int=None,
    max_train_token: int=None,
    process_group_list: List[Any] = None,
    offset: int=0, packing: bool=False
    ):
    assert micro_batch_size is not None or max_train_token is not None, \
        "At least one of micro_batch_size or max_train_token should be specified"
    if packing:
        # Using bin packing to slice minibatch
        return regroup_data_packing(data_list, process_group_list, max_train_token, offset)
    else:
        # Calculate num_micro_batch, add 1 to avoid dropping data
        # Slice data_list evenly with micro batch size
        mini_batch_size = len(data_list)
        num_micro_batch = math.ceil(mini_batch_size / micro_batch_size)
        micro_batches = [[] for _ in range(num_micro_batch)]
        for i in range(num_micro_batch):
            start_idx = i * micro_batch_size
            end_idx = min((i+1) * micro_batch_size, mini_batch_size)
            for sample_id in range(start_idx, end_idx):
                data_list[sample_id].update({'id_in_list': sample_id})
                micro_batches[i].append(data_list[sample_id])

        return micro_batches

def padding_tensor(tensor_list):
    dim_size = len(tensor_list[0].shape)
    seqlen = max(tensor.shape[0] for tensor in tensor_list)
    pad_size = [seqlen - tensor.shape[0] for tensor in tensor_list]
    # Right pad tensor on dim 0
    batched_tensor = [
        F.pad(tensor_list[i], (0, 0) * (dim_size - 1) + (0, pad_size[i]))
        for i in range(len(tensor_list))
    ]
    return torch.stack(batched_tensor)

def batching(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(data_list) == 0:
        return None
    batched_data = defaultdict(list)
    for key in data_list[0]:
        batched_data[key] = [data[key] for data in data_list]
        if key in ['pixel_values', 'image_grid_thw', 'rope_deltas']:
            batched_data[key] = torch.cat(batched_data[key], dim=0)
        elif isinstance(batched_data[key][0], torch.Tensor):
            batched_data[key] = padding_tensor(batched_data[key])
    return batched_data

def split_and_unpadding(input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
    valid_seq = torch.sum(attention_mask, dim=-1).cpu().tolist()
    tensor_list = [
        input_tensor[i, :valid_seq[i]] for i in range(input_tensor.shape[0])
    ]
    return tensor_list

def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    if len(hidden_states.shape)>3:
        # ==============================================================================
        # patch for vl position_ids to merge the channel dimension
        channel_results = []
        for channel in range(hidden_states.shape[0]):
            indexed = index_first_axis(rearrange(hidden_states[channel], "b s ... -> (b s) ..."), indices)
            channel_results.append(indexed)
        return (
            torch.stack(channel_results),
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
        )
        # ==============================================================================
    else:
        return (
            index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
        )
