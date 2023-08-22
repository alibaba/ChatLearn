# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
"""Forward step utilities."""

import torch

from megatron import get_args
from megatron.core import mpu
from megatron.text_generation.communication import send_to_next_pipeline_rank, recv_from_prev_pipeline_rank_
from megatron.text_generation.forward_step import _allocate_recv_buffer


def forward_step_helper(model, tokens, position_ids, attention_mask, pooling_sequence_index=None, pooling=False):
    # Pipelining case.
    args = get_args()
    if args.pipeline_model_parallel_size > 1:
        current_batch_x_seqlen = tokens.size(0) * tokens.size(1)
        if current_batch_x_seqlen >= args.inference_batch_times_seqlen_threshold:
            micro_batch_size = \
                max(1, args.inference_batch_times_seqlen_threshold // tokens.size(1))
            return _with_pipelining_forward_step(model,
                                                 tokens,
                                                 position_ids,
                                                 attention_mask,
                                                 micro_batch_size,
                                                 pooling_sequence_index=pooling_sequence_index,
                                                 pooling=pooling)

    return _no_pipelining_forward_step(model,
                                       tokens,
                                       position_ids,
                                       attention_mask,
                                       pooling_sequence_index=pooling_sequence_index)


def _forward_step_helper(model, tokens, position_ids, attention_mask,
                         recv_buffer=None, pooling_sequence_index=None):
    """Single forward step. Update the allocate memory flag so
    only the first time the memory is allocated."""
    batch_size = tokens.size(0)
    sequence_length = tokens.size(1)
    if recv_buffer is None:
        recv_buffer = _allocate_recv_buffer(batch_size, sequence_length)

    # Receive from previous stage.
    recv_from_prev_pipeline_rank_(recv_buffer)

    # Forward pass through the model.
    model.set_input_tensor(recv_buffer)
    if pooling_sequence_index is not None:
        output_tensor = model(tokens, position_ids, attention_mask, pooling_sequence_index=pooling_sequence_index)
    else:
        output_tensor = model(tokens, position_ids, attention_mask)

    # Send output to the next stage.
    send_to_next_pipeline_rank(output_tensor)

    return output_tensor


def _no_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                recv_buffer=None, pooling_sequence_index=None):
    """If recv_buffer is none, we will allocate one on the fly."""
    # Run a simple forward pass.
    output_tensor = _forward_step_helper(model, tokens, position_ids,
                                         attention_mask, recv_buffer=recv_buffer,
                                         pooling_sequence_index=pooling_sequence_index)

    logits = None
    if mpu.is_pipeline_last_stage():
        logits = output_tensor

    return logits


def _with_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                  micro_batch_size, pooling_sequence_index=None, pooling=False):
    """No interleaving is supported."""
    sequence_length = tokens.size(1)
    batch_size = tokens.size(0)

    # Divide the batch dimension into micro batches.
    num_micro_batches, last_chunk = divmod(batch_size,
                                           micro_batch_size)
    if last_chunk > 0:
        num_micro_batches += 1

    # Preallocate memory for output logits.
    logits = None
    if mpu.is_pipeline_last_stage():
        args = get_args()
        if pooling:
            logits = None
        else:
            if args.parallel_output:
                padded_vocab_size = args.padded_vocab_size // args.tensor_model_parallel_size
            else:
                padded_vocab_size = args.padded_vocab_size
            logits = torch.empty(
                (batch_size, sequence_length, padded_vocab_size),
                dtype=torch.float32, device=torch.cuda.current_device())

    # Preallocate recv buffer.
    recv_buffer = _allocate_recv_buffer(micro_batch_size, sequence_length)

    for micro_batch_index in range(num_micro_batches):
        # Slice among the batch dimenion.
        start = micro_batch_index * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        this_micro_batch_size = end - start
        tokens2use = tokens[start:end, ...]
        position_ids2use = position_ids[start:end, ...]

        # Run a simple forward pass.
        if this_micro_batch_size != micro_batch_size:
            recv_buffer = None
        output = _forward_step_helper(model, tokens2use, position_ids2use,
                                      attention_mask, recv_buffer=recv_buffer,
                                      pooling_sequence_index=pooling_sequence_index)

        # Copy logits.
        if mpu.is_pipeline_last_stage():
            if logits is None:
                logits = torch.empty([batch_size] + list(output.shape)[1:],
                                     dtype=torch.float32, device=torch.cuda.current_device())
            logits[start:end, ...] = output

    return logits