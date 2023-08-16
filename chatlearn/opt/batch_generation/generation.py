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
"generation function."

import importlib.util
import torch

import chatlearn
from .forward_step import ForwardStep

megatron_exist = importlib.util.find_spec("megatron")
if megatron_exist:
    from megatron import get_args, get_tokenizer
    from megatron.core import mpu
    from megatron.text_generation.communication import (
        copy_from_last_to_first_pipeline_stage,
        broadcast_from_last_pipeline_stage,
        broadcast_from_last_to_first_pipeline_stage)
    from megatron.text_generation.generation import _build_attention_mask_and_position_ids


def generate_tokens_probs_and_return_on_first_stage(
        model, tokens, lengths,
        return_output_log_probs=False,
        top_k=0, top_p=0.0, top_p_decay=0.0, top_p_bound=0.0,
        temperature=1.0,
        use_eod_token_for_early_termination=True,
        stop_on_double_eol=False,
        stop_on_eol=False,
        prevent_newline_after_colon=True
        ):
    """Main token generation function. 
    Equals to megatron.text_generation.generation.generate_tokens_probs_and_return_on_first_stage
    Arguments:
        model: no interleaving is supported.
        tokens: prompt tokens extended to be of size [b, max-sequence-length]
        lengths: original prompt length, size: [b]
        return_output_log_probs: flag to calculate the log probability of
            the generated tokens. Note that the log probability is the one
            from the original logit.
        top_k, top_p: top-k and top-p sampling parameters.
            Note that top-k = 1 is gready. Also, these paramters are
            exclusive meaning that:
                if top-k > 0 then we expect top-p=0.
                if top-p > 0 then we check for top-k=0.
        temperature: sampling temperature.
        use_eod_token_for_early_termination: if True, do early termination if
            all the sequences have reached this token.
        prevent_newline_after_colon: if True, it will disable generating new line \n after :
    Note: Outside of model, other parameters only need to be available on
          rank 0.
    Outputs: Note that is size is adjusted to a lower value than
             max-sequence-length if generation is terminated early.
        tokens: prompt and generated tokens. size: [b, :]
        generated_sequence_lengths: total length (including prompt) of
            the generated sequence. size: [b]
        output_log_probs: log probability of the selected tokens. size: [b, s]
    """
    args = get_args()
    tokenizer = get_tokenizer()

    batch_size = tokens.size(0)
    min_prompt_length = lengths.min().item()
    if chatlearn.get_args().active_module_args.batch_generation.min_prompt_length:
        min_prompt_length = min(min_prompt_length, chatlearn.get_args().active_module_args.batch_generation.min_prompt_length)
    max_sequence_length = tokens.size(1)

    if max_sequence_length > args.max_position_embeddings:
        raise ValueError("Length of prompt + tokens_to_generate longer than allowed")

    if max_sequence_length * batch_size > args.max_tokens_to_oom:
        raise ValueError("Too many tokens.  " + str(max_sequence_length*batch_size)+ " is greater than "+str(args.max_tokens_to_oom))

    # forward step.
    forward_step = ForwardStep(model, batch_size, max_sequence_length)

    # Added termination_id to support the case that we want to terminate the
    # generation once that id is generated.
    if hasattr(args, 'eos_id'):
        termination_id = args.eos_id
    else:
        termination_id = tokenizer.eod

    # ===================
    # Pre-allocate memory
    # ===================

    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_sequence_length - 1)
    # Lengths of generated seuquence including including prompts.
    generated_sequence_lengths = None
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = torch.empty(output_log_probs_size,
                                           dtype=torch.float32,
                                           device=torch.cuda.current_device())
        generated_sequence_lengths = torch.ones(
                batch_size, dtype=torch.int64,
                device=torch.cuda.current_device()) * max_sequence_length

    # Whether we have reached a termination id.
    is_generation_done = torch.zeros(batch_size, dtype=torch.uint8,
                                     device=torch.cuda.current_device())

    # =============
    # Run infernece
    # =============

    with torch.no_grad():
        attention_mask, position_ids = _build_attention_mask_and_position_ids(
            tokens)
        prev_context_length = 0
        for context_length in range(min_prompt_length, max_sequence_length):

            # Pick the slice that we need to pass through the network.
            tokens2use = tokens[:, prev_context_length:context_length]
            positions2use = position_ids[:, prev_context_length:context_length]
            attention_mask2use = attention_mask[
                ..., prev_context_length:context_length, :context_length]

            # logits will be meanigful only in the last pipeline stage.
            kwargs = {}
            kwargs["top_k"] = top_k
            kwargs["top_p"] = top_p
            kwargs["prevent_newline_after_colon"] = prevent_newline_after_colon
            kwargs["temperature"] = temperature
            kwargs["lengths"] = lengths
            kwargs["context_length"] = context_length
            kwargs["return_output_log_probs"] = return_output_log_probs
            kwargs["prev_context_length"] = prev_context_length
            output_log_probs, new_sample, started = \
                forward_step(tokens2use, positions2use, attention_mask2use, tokens, output_log_probs, tokenizer, **kwargs)


            if top_p > 0.0 and top_p_decay > 0.0:
                top_p = top_p * top_p_decay
                if top_p_bound > 0.0:
                    top_p = max(top_p, top_p_bound)

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(batch_size, torch.int64,
                                                   tokens[:, context_length])

            # Update the context length for the next token generation.
            prev_context_length = context_length

            # Check if all the sequences have hit the termination_id.
            done = None
            if mpu.is_pipeline_last_stage():
                # TODO(rprenger) These stopping methods are tokenizer dependent
                # instead tokenization should be in the inference loop so stop sequences can be used
                if stop_on_double_eol:
                    hit_double_eol = (new_sample == 628).byte() & started.byte()
                    hit_two_eols = (new_sample == 198).byte() & (tokens[:, context_length-1] == 198).byte() & started.byte()
                    done_token = hit_double_eol | hit_two_eols
                elif stop_on_eol:
                    hit_double_eol = (new_sample == 628).byte() & started.byte()
                    hit_eol = (new_sample == 198).byte() & started.byte()
                    done_token = hit_double_eol | hit_eol
                else:
                    done_token = (new_sample == termination_id).byte() & \
                        started.byte()

                just_finished = (done_token & ~is_generation_done).bool()
                generated_sequence_lengths[just_finished.view(-1)] = \
                    context_length + 1
                is_generation_done = is_generation_done | done_token
                done = torch.all(is_generation_done)
            done = broadcast_from_last_pipeline_stage(1, torch.uint8,
                                                      tensor=done)
            if use_eod_token_for_early_termination and done:
                break

    # ===================================================
    # Update the length of based on max generated length.
    # ===================================================

    tokens = tokens[:, :(context_length + 1)]
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = output_log_probs[:, :context_length]

    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================

    generated_sequence_lengths = broadcast_from_last_to_first_pipeline_stage(
        batch_size, torch.int64, generated_sequence_lengths)
    if return_output_log_probs:
        output_log_probs_size = (batch_size, context_length)
        output_log_probs = broadcast_from_last_to_first_pipeline_stage(
            output_log_probs_size, torch.float32, output_log_probs)

    return tokens, generated_sequence_lengths, output_log_probs
