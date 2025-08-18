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

import torch
import torch.nn.functional as F

def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy

def entropy_from_logits_with_chunking(logits: torch.Tensor, chunk_size: int = 128):
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

def generate_loss_mask_position_ids(tokens: torch.Tensor, prompt_token_length: List[int], response_token_length: List[int], prompt_position_ids: List[List[int]]):
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

    if len(prompt_position_ids)>0 and len(prompt_position_ids[0])==0:
        # text only
        position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)
    else:
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
    return attn_mask, loss_mask, position_ids
