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
"forward step."

from collections.abc import Iterable

import importlib.util
import torch
import torch.nn.functional as F

import chatlearn

megatron_exist = importlib.util.find_spec("megatron")
if megatron_exist:
    from megatron import get_args
    from megatron.core import mpu
    from megatron.text_generation.forward_step import InferenceParams
    from megatron.text_generation.forward_step import _forward_step_helper
    from megatron.text_generation.sampling import sample


def quantify_micro_batch(batch_size, sequence_length):
    """Calculate micro-batch-size and num-micro-batch for batch generation pipeline."""
    num_max_tokens = chatlearn.get_args().active_module_args.batch_generation.num_max_tokens
    if num_max_tokens is None:
        num_max_tokens = 0
    micro_batch_size = int(num_max_tokens // sequence_length)
    micro_batch_size = min(
        micro_batch_size,
        batch_size) if micro_batch_size else batch_size

    # Divide the batch dimension into micro batches.
    num_micro_batches, last_chunk = divmod(batch_size,
                                           micro_batch_size)
    if last_chunk > 0:
        num_micro_batches += 1

    return micro_batch_size, num_micro_batches


def _forward_step(model, tokens, position_ids, attention_mask, inference_params,
        original_tokens, output_log_probs, tokenizer, **kwargs):
    """No interleaving is supported."""
    sequence_length = tokens.size(1)
    batch_size = tokens.size(0)
    micro_batch_size, num_micro_batches = quantify_micro_batch(batch_size, sequence_length)

    # Preallocate memory for output logits.
    logits = None
    args = get_args()

    top_k = kwargs.get("top_k")
    top_p = kwargs.get("top_p")
    prevent_newline_after_colon = kwargs.get("prevent_newline_after_colon")
    temperature = kwargs.get("temperature")
    lengths = kwargs.get("lengths")
    context_length = kwargs.get("context_length")
    return_output_log_probs = kwargs.get("return_output_log_probs")
    prev_context_length = kwargs.get("prev_context_length")

    new_sample_ret = None
    started_ret = None

    for micro_batch_index in range(num_micro_batches):
        # Slice among the batch dimenion.
        start = micro_batch_index * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        this_micro_batch_size = end - start
        logits = torch.empty(
            (this_micro_batch_size, sequence_length, args.padded_vocab_size),
            dtype=torch.float32, device=torch.cuda.current_device())
        tokens2use = tokens[start:end, ...]
        position_ids2use = position_ids[start:end, ...]
        lengths_use = lengths[start:end]
        tokens_orig = original_tokens[start:end, ...]

        # Run a simple forward pass.
        logits = _forward_step_helper(model, tokens2use, position_ids2use,
                                      attention_mask, inference_params,
                                      recv_buffer=None)

        # Adjust the batch size offset to account for the micro-batch.
        inference_params.batch_size_offset += this_micro_batch_size

        if mpu.is_pipeline_last_stage():
            if prevent_newline_after_colon:
                logits[tokens2use[:, -1] == tokenizer.tokenize(':')[0], -1, tokenizer.tokenize('\n')[0]] = -1e10 # disable "\n" after ":"
            # Always the last stage should have an output.
            assert logits is not None

            if get_args().numerical_stable:
                logits_max = torch.max(logits, dim=-1)[0]
                logits_min = torch.min(logits, dim=-1)[0]
                logits_normalize = (logits_min + logits_max) / 2
                logits.sub_(logits_normalize.unsqueeze(dim=-1))

            # Sample.
            last_token_logits = logits[:, -1, :]
            new_sample = sample(last_token_logits,
                                top_k=top_k,
                                top_p=top_p,
                                temperature=temperature,
                                vocab_size=tokenizer.vocab_size)

            # If a prompt length is smaller or equal th current context
            # length, it means we have started generating tokens
            started = lengths_use <= context_length
            if new_sample_ret is None:
                new_sample_ret = new_sample
                started_ret = started
            else:
                new_sample_ret = torch.cat((new_sample_ret, new_sample), dim=0)
                started_ret = torch.cat((started_ret, started), dim=0)
            # Update the tokens.
            tokens_orig[started, context_length] = new_sample[started]

            # Calculate the log probabilities.
            if return_output_log_probs:
                log_probs = F.log_softmax(logits, dim=2)
                if return_output_log_probs:
                    # Pick the tokens that we need to get the log
                    # probabilities for. Note that next input token is
                    # the token which we selected in the current logits,
                    # so shift by 1.
                    indices = torch.unsqueeze(
                        tokens_orig[
                            :,
                            (prev_context_length + 1):(context_length + 1)],
                        2)
                    output_log_probs[start:end,
                                    prev_context_length:context_length] = \
                        torch.gather(log_probs, 2, indices).squeeze(2)

    # Once we are done with all the micro-batches, we can
    # adjust the sequence length offset.
    inference_params.sequence_len_offset += sequence_length
    # and reset the batch size offset
    inference_params.batch_size_offset = 0

    return output_log_probs, new_sample_ret, started_ret


class ForwardStep:
    """Forward step function with all the communications.
    We use a class here to hide the inference parameters
    from the outside caller."""

    def __init__(self, model, max_batch_size, max_sequence_len):
        """Set values so we don't need to do it multiple times."""
        # Make sure model is in eval mode.
        assert not isinstance(model, Iterable), \
            'interleaving schedule is not supported for inference'
        model.eval()
        self.model = model
        # Initialize inference parameters.
        self.inference_params = InferenceParams(max_batch_size,
                                                max_sequence_len)
        # Pipelining arguments.
        args = get_args()
        self.pipeline_size_larger_than_one = (
            args.pipeline_model_parallel_size > 1)
        # Threshold of pipelining.
        self.pipelining_batch_x_seqlen = \
            args.inference_batch_times_seqlen_threshold


    def __call__(self, tokens, position_ids, attention_mask, original_tokens, output_log_probs, tokenizer, **kwargs):
        """Invocation of the forward methods. Note that self.inference_params
        is being modified by the forward step."""
        # Pipelining case.
        if self.pipeline_size_larger_than_one:
            if chatlearn.get_args().active_module_args.batch_generation.ranking:
                raise RuntimeError("Pipeline not support for batch generation ranking")

        return _forward_step(self.model,
                             tokens,
                             position_ids,
                             attention_mask,
                             self.inference_params,
                             original_tokens,
                             output_log_probs,
                             tokenizer,
                             **kwargs)
