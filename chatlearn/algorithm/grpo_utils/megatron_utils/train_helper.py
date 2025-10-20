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
"""helper function for megatron forward, including preprocessing and loss helper"""
import copy
from typing import Dict, Iterator, Union, List, Any
from functools import partial

import torch
from torch import distributed as dist
import torch.nn.functional as F

import transformer_engine_torch as tex
from flash_attn.bert_padding import pad_input, unpad_input

from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import (
    get_args, get_timers,
    is_last_rank,
    get_tokenizer,
    print_rank_last
)
from megatron.training.training import print_datetime
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    report_memory,
    unwrap_model
)

from chatlearn.utils import to_device

from ..loss_gallery import calculate_grpo_loss, calculate_gspo_loss

# TODO: simplify this function
def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
    stats,
    more_grad_norm,
    name,
    metric_list=None,
):
    """Log training information such as losses, timing, ...."""
    # pylint: disable=unused-argument, unused-variable
    args = get_args()
    timers = get_timers()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = (
            total_loss_dict.get(advanced_iters_key, 0) + 1
        )
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = (
        total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    )
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = (
                total_loss_dict.get(key, torch.tensor([0.0], dtype=torch.float32, device='cuda')) + loss_dict[key]
            )
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value # pylint: disable=comparison-with-itself
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(
        got_nan
    )

    # Calculate batch size.
    batch_size = (
        args.micro_batch_size * args.data_parallel_size * get_num_microbatches()
    )

    total_iterations = (
        total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]
    )

    iter_dict = {}
    consumed_train_samples_dict = {}
    # Tensorboard values.
    if is_last_rank():

        for key in loss_dict:
            iter_dict[f"{key}"] = loss_dict[key]
            consumed_train_samples_dict[key + " vs samples"] = loss_dict[
                key
            ]

        if grad_norm is not None:
            iter_dict["grad_norm"] = grad_norm
            consumed_train_samples_dict["grad-norm vs samples"] = grad_norm

        if more_grad_norm is not None:
            for k in more_grad_norm:
                iter_dict[f"{k}" + " grad_norm"] = more_grad_norm[k]
                consumed_train_samples_dict[f"{k}" + " grad-norm vs samples"] = (
                    more_grad_norm[k]
                )

        if params_norm is not None:
            iter_dict["params-norm"] = params_norm
            consumed_train_samples_dict["params-norm vs samples"] = (
                params_norm
            )

    elapsed_time = 0
    elapsed_time_per_iteration = elapsed_time / total_iterations
    if args.log_timers_to_tensorboard:
        iter_dict["iteration-time"] = elapsed_time_per_iteration

    log_string = " iteration {:8d}/infinity |".format(iteration)
    log_string += " consumed samples: {:12d} |".format(args.consumed_train_samples)
    log_string += " elapsed time per iteration (ms): {:.1f} |".format(
        elapsed_time_per_iteration * 1000.0
    )
    log_string += " learning rate: {:.3E} |".format(learning_rate)
    log_string += " global batch size: {:5d} |".format(batch_size)

    for key in total_loss_dict:
        if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
            avg = total_loss_dict[key].item() / float(
                max(1, total_loss_dict[advanced_iters_key])
            )
            log_string += " {}: {:.6E} |".format(key, avg)

            total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
    log_string += " loss scale: {:.1f} |".format(loss_scale)

    if grad_norm is not None:
        log_string += " grad norm: {:.3f} |".format(grad_norm)

    if more_grad_norm is not None:
        for k in more_grad_norm:
            log_string += "{} grad norm: {:.3f} |".format(k, more_grad_norm[k])

    log_string += " number of nan iterations: {:3d} |".format(
        total_loss_dict[nan_iters_key]
    )
    total_loss_dict[advanced_iters_key] = 0
    total_loss_dict[skipped_iters_key] = 0
    total_loss_dict[nan_iters_key] = 0
    print_rank_last(log_string)
    if report_memory_flag and learning_rate > 0.0:
        report_memory("(after {} iterations)".format(iteration))
        report_memory_flag = False
    print_datetime("Logger")

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1
    ):
        wandb_dicts = {}
        wandb_dicts.update(stats)
        wandb_dicts.update(iter_dict)
        wandb_dict_copy = copy.deepcopy(wandb_dicts)
        if metric_list is None:
            metric_list = [wandb_dict_copy]
        else:
            metric_list.append(wandb_dict_copy)

    return report_memory_flag

def _reduce_from_tensor_model_parallel_region(
    output_on_this_cp_rank: torch.Tensor,
    seq_indices: torch.Tensor
):
    """
    SHAPE:
        output_on_this_cp_rank: [mbs, length, *] or [1, total_nnz, *]
        seq_indices: [mbs, length] or [1, total_nnz]
    """
    cp_size = mpu.get_context_parallel_world_size()
    cp_group = mpu.get_context_parallel_group()
    mbs, length = output_on_this_cp_rank.shape[:2]
    output = torch.zeros(mbs, length * cp_size, *output_on_this_cp_rank.shape[2:], device=output_on_this_cp_rank.device)
    if output.ndim > 2:
        seq_indices = seq_indices.view((mbs, length,) + (1,) * (output.ndim - 2) )
    output.scatter_(1, seq_indices.to(torch.int64), output_on_this_cp_rank)
    reduce_from_tensor_model_parallel_region(output, group=cp_group)
    return output

def reduce_from_context_parallel_region(
    logprobs: torch.Tensor,
    is_packing: bool = False,
    inputs: Dict[str, Any] = None
):
    if mpu.get_context_parallel_world_size() > 1:
        logprobs = _reduce_from_tensor_model_parallel_region(logprobs, inputs['seq_indices'])

    if is_packing:
        logprobs = pad_input(
            logprobs.permute(1, 0),
            inputs['indices'],
            inputs['ori_batch_size'],
            inputs['ori_seq_len'],
        ).squeeze(-1)

    logprobs = logprobs[:, :logprobs.shape[1] - inputs['pad_size']]
    return logprobs


def get_batch(
    data_iter: Iterator[Dict[str, Union[torch.Tensor, List[Any]]]],
    is_training: bool,
    is_packing: bool = False
) -> Dict[str, Any]:
    """Build the input data for MCore model based on the raw batch
    produced by data_iter. If is_training is False, the raw batch
    should have `all_tokens`; otherwise, the raw batch should
    also have `prompt_token_length`, `response_token_length`,
    `ref_logprobs`, `old_logprobs` and `advantages`.

    Args:
        data_iter (Iterator[Dict[str, Union[torch.Tensor, List[Any]]]]):
        The iterator produced raw batch.
        is_training (bool): Whether called in training mode.
    """
    args = get_args()
    data_b = next(data_iter)

    tokens = data_b["all_tokens"].long().cuda() # shape: [mbs, seq_length]
    mbs, seqlen = tokens.shape
    prompt_token_length = to_device("cuda", data_b["prompt_token_length"])
    response_token_length = to_device("cuda", data_b["response_token_length"])

    if is_packing:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_group().rank()
        align_size = tp_size * cp_size * 2
        attn_mask = torch.zeros_like(tokens, dtype=torch.int32)
        for i, (prompt_length, response_length) in enumerate(
            zip(prompt_token_length, response_token_length)
        ):
            attn_mask[i, : prompt_length + response_length] = 1

        seqlens_in_batch = attn_mask.sum(dim=-1, dtype=torch.int32)
        pad_size = (align_size - seqlens_in_batch % align_size) % align_size
        seqlens_in_batch_padded = seqlens_in_batch + pad_size

        tokens_padded = torch.full((mbs, seqlens_in_batch_padded.max()), get_tokenizer().eod, dtype=tokens.dtype, device=tokens.device)
        attn_mask_for_padding = torch.zeros((mbs, seqlens_in_batch_padded.max()), dtype=tokens.dtype, device=tokens.device)
        for i in range(mbs):
            seqlen_in_batch = seqlens_in_batch[i]
            seqlen_in_batch_padded = seqlens_in_batch_padded[i]
            tokens_padded[i, :seqlen_in_batch] = tokens[i, :seqlen_in_batch]
            attn_mask_for_padding[i, :seqlen_in_batch_padded] = 1

        (
            tokens,
            indices,
            cu_seqlens_padded,
            max_seqlen_in_batch,
            *_
        ) = unpad_input(tokens_padded.unsqueeze(-1), attn_mask_for_padding)

        seq_indices = tex.thd_get_partitioned_indices(
            cu_seqlens_padded, tokens.shape[0], cp_size, cp_rank
        )
        tokens_on_this_cp_rank = tokens.index_select(0, seq_indices)
        labels = torch.roll(tokens, shifts=-1, dims=0)
        labels_on_this_cp_rank = labels.index_select(0, seq_indices)

        loss_mask_for_unpadding = torch.zeros_like(tokens, dtype=torch.int32, device=tokens.device)
        for i, (prompt_length, response_length) in enumerate(
            zip(prompt_token_length, response_token_length)
        ):
            start_idx = cu_seqlens_padded[i]
            loss_mask_for_unpadding[start_idx + prompt_length: start_idx + prompt_length + response_length, 0] = 1

        loss_mask_for_unpadding = torch.roll(loss_mask_for_unpadding, shifts=-1, dims=0)
        loss_mask_on_this_cp_rank = loss_mask_for_unpadding.index_select(0, seq_indices)
        num_tokens_on_this_cp_rank = loss_mask_on_this_cp_rank.sum()

        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            max_seqlen_q=max_seqlen_in_batch,
            max_seqlen_kv=max_seqlen_in_batch
        )

        input_data = {
            "all_tokens": tokens_on_this_cp_rank.transpose(0, 1),
            "all_token_attention_mask": None,
            "all_token_loss_mask": None,
            "all_token_position_ids": None,
            "labels": labels_on_this_cp_rank.transpose(0, 1),
            'packed_seq_params': packed_seq_params,
            "ori_seq_len": seqlens_in_batch_padded.max(),
            "ori_batch_size": mbs,
            "seq_indices": seq_indices.unsqueeze(0), # [1, total_nnz_per_cp_rank]
            "indices": indices,
            "num_tokens_on_this_cp_rank": num_tokens_on_this_cp_rank,
            'pad_size': seqlens_in_batch_padded.max() - seqlen
        }

    else:
        pad_size = 0
        pad_token = get_tokenizer().eod
        if not args.variable_seq_lengths:
            pad_size = args.seq_length - tokens.shape[1]
        else:
            divisor = mpu.get_tensor_model_parallel_world_size()
            total_nnz = tokens.shape[1]
            pad_size = (divisor - total_nnz % divisor) % divisor

        position_ids = data_b.get('position_ids', None)
        tokens = F.pad(tokens, (0, pad_size), value=pad_token)
        labels = torch.roll(tokens, shifts=-1, dims=1)
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            get_tokenizer().eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss,
            position_ids
        )
        if mpu.get_context_parallel_world_size() == 1:
            input_data = {
                "all_tokens": tokens,
                "all_token_attention_mask": attention_mask,
                "all_token_position_ids": position_ids,
                "labels": labels,
                'packed_seq_params': None,
                "pad_size": pad_size,
            }
        elif mpu.get_context_parallel_world_size() >1:

            loss_mask= torch.zeros_like(tokens, dtype=torch.int32, device=tokens.device)
            for i, (prompt_length, response_length) in enumerate(
                zip(prompt_token_length, response_token_length)
            ):
                loss_mask[i, prompt_length: prompt_length + response_length] = 1

            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1)

            input_batch = {
                "all_tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            chunked_dataset = get_batch_on_this_cp_rank(input_batch)
            input_data = {
                "all_tokens": chunked_dataset['all_tokens'],
                "all_token_attention_mask": chunked_dataset['attention_mask'],
                "all_token_position_ids": chunked_dataset['position_ids'],
                "labels": chunked_dataset['labels'],
                'packed_seq_params': None,
                "ori_seq_len": None,
                "ori_batch_size": None,
                "seq_indices": chunked_dataset['position_ids'].clone(),
                "indices": None,
                "pad_size": pad_size,
                "num_tokens_on_this_cp_rank": chunked_dataset['loss_mask'].sum(),
            }

    if 'pixel_values' in data_b.keys():
        # vl
        input_data.update(
            {
                "pixel_values": data_b['pixel_values'], # [token_length, token_num]
                "image_grid_thw": data_b['image_grid_thw'], # [batch_size, 3]
                "rope_deltas": data_b['rope_deltas'], # [batch_size, 1]
                "image_input_mask": tokens==get_tokenizer().vocab['<|image_pad|>'] # [batch_size, token_length]
            }
        )

    if is_training:
        ref_logprobs = data_b["ref_logprobs"].float()
        old_logprobs = data_b["old_logprobs"].float()
        advantages = data_b["advantages"]
        if "loss_mask" in data_b:
            loss_mask = data_b['loss_mask']
        else:
            loss_mask = torch.zeros_like(data_b["all_tokens"], dtype=torch.int32)
            for i, (prompt_length, response_length) in enumerate(
                zip(prompt_token_length, response_token_length)
            ):
                loss_mask[i, prompt_length: prompt_length + response_length] = 1
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1)

        input_data.update({
            "all_token_loss_mask": loss_mask,
            "advantages": advantages,
            "ref_logprobs": ref_logprobs,
            "old_logprobs": old_logprobs
        })

    for k, v in input_data.items():
        input_data[k] = to_device("cuda", v)
    return input_data

def loss_func(
    inputs: Dict,
    losses: Dict[str, torch.Tensor]
):
    """Loss function.

    Args:
        inputs (Dict): The full inputs of this micro-batch.
        losses (Dict[str, torch.Tensor]): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    require_bp_keys = ['pg_loss']

    loss_mask = inputs["all_token_loss_mask"].float()
    total_loss_for_bp = 0
    reporting_losses = {}
    for key, loss in losses.items():
        if key not in require_bp_keys:
            loss = loss.detach()
        if key.endswith('_sample_average'):
            final_loss = (loss.float() * loss_mask).sum() / (1e-5 + loss_mask.sum())
        else:
            final_loss = (loss.float() * loss_mask).sum()
        if key in require_bp_keys:
            total_loss_for_bp = total_loss_for_bp + final_loss

        reporting_losses[key] = final_loss.detach().clone().to(torch.float)

    num_tokens = inputs.get('num_tokens_on_this_cp_rank', loss_mask.sum().clone().detach()).to(torch.int)
    reporting_losses["num_tokens"] = num_tokens
    reporting_losses["num_samples"] = torch.ones_like(num_tokens)

    return total_loss_for_bp, num_tokens, reporting_losses

def _compute_all_losses(
    module_args,
    model,
    all_token_logits: torch.Tensor,
    labels: torch.Tensor,
    training_inputs: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Compute all required losses.
    
    Args:
        all_token_logits (torch.Tensor): logits of input tokens. shape: [s, b, h] or [total_nnz, 1, h]
        labels (torch.Tensor): labels of input tokens. shape: [s, b] or [total_nnz, 1]
        training_inputs (Dict[str, Any]): All training inputs.
    
    """
    forward_logprob = (
        unwrap_model(model).compute_language_model_loss(labels, all_token_logits.clone()) * -1
    )

    forward_logprob = reduce_from_context_parallel_region(forward_logprob, module_args.packing, training_inputs)

    old_logprobs = training_inputs["old_logprobs"]
    ref_logprobs = training_inputs["ref_logprobs"]
    advantages = training_inputs["advantages"]


    if module_args.use_group_sequence_policy:
        (
            pg_loss,
            is_positive_clipped,
            is_negative_clipped,
            is_clipped,
        ) = calculate_gspo_loss(
            log_probs=forward_logprob,
            old_log_probs=old_logprobs,
            advantages=advantages,
            diff_clip_ratio=module_args.diff_clip_ratio,
            pos_clip_ratio=module_args.pos_clip_ratio,
            neg_clip_ratio=module_args.neg_clip_ratio,
            final_clip_ratio=module_args.final_clip_ratio,
            loss_mask = training_inputs['all_token_loss_mask']
        )
    else:
        pg_loss = calculate_grpo_loss(
            log_probs=forward_logprob,
            old_log_probs=old_logprobs,
            advantages=advantages,
            diff_clip_ratio=module_args.diff_clip_ratio,
            pos_clip_ratio=module_args.pos_clip_ratio,
            neg_clip_ratio=module_args.neg_clip_ratio,
            final_clip_ratio=module_args.final_clip_ratio
        )

    entropy_loss = entropy_from_tensor_parallel_logits(all_token_logits).permute(1, 0)

    entropy_loss = reduce_from_context_parallel_region(entropy_loss, module_args.packing, training_inputs)

    kl = ref_logprobs - forward_logprob
    ratio = torch.exp(kl)
    ratio[~training_inputs['all_token_loss_mask'].bool()] = 1
    assert not torch.isinf(ratio).any(), "kl loss ratio has inf values"
    assert not torch.isnan(ratio).any(), "kl loss ratio has nan values"
    kld = (ratio - kl - 1).contiguous()
    kl_loss = torch.clamp(kld, min=-10, max=10)

    if module_args.use_group_sequence_policy:
        return {
            'pg_loss': pg_loss,
            'entropy_loss': entropy_loss,
            'kl_loss': kl_loss,
            'is_positive_clipped': is_positive_clipped,
            'is_negative_clipped': is_negative_clipped,
            'is_clipped': is_clipped,
            'is_positive_clipped_sample_average': is_positive_clipped,
            'is_negative_clipped_sample_average': is_negative_clipped,
            'is_clipped_sample_average': is_clipped,
        }
    else:
        return {
            'pg_loss': pg_loss,
            'entropy_loss': entropy_loss,
            'kl_loss': kl_loss
        }


def forward_step(data_iterator, model, *, is_training: bool=False, is_packing: bool=False, module_args = None):
    """Forward step."""

    inputs = get_batch(
        data_iterator,
        is_training=is_training,
        is_packing=is_packing
    )

    kwargs = {
        'input_ids': inputs["all_tokens"],
        'position_ids': inputs["all_token_position_ids"],
        'labels': inputs["labels"] if not is_training else None,
    }

    if is_packing:
        kwargs.update({'packed_seq_params': inputs['packed_seq_params']})

    if 'pixel_values' in inputs:
        kwargs.update({
            'vision_data': inputs["pixel_values"],
            'vision_grid_thw': inputs["image_grid_thw"],
            'image_input_mask': inputs["image_input_mask"],
            'video_start_index': inputs["image_input_mask"].sum().cpu().item()
        })
    else:
        kwargs.update({
            'attention_mask': inputs["all_token_attention_mask"]
        })

    # NOTE:
    # 1) when post_process is False, model returns hidden states
    # 2) when post_process is True:
    #   1) if is_training is False, model returns logprobs
    #   2) otherwise, model returns logits and loss should be computed by `_compute_all_losses`
    output_tensor = model(**kwargs)

    if unwrap_model(model).post_process:
        if is_training:
            output_tensor = _compute_all_losses(
                module_args=module_args,
                model=model,
                all_token_logits=output_tensor.transpose(0, 1).contiguous(),
                labels=inputs['labels'],
                training_inputs=inputs
            )
        else:
            output_tensor = reduce_from_context_parallel_region(output_tensor, is_packing, inputs)

    # NOTE: just returns the output tensor (the first argument).
    wrapped_loss_func = lambda x, **_: x # pylint: disable=unnecessary-lambda-assignment
    if is_training:
        wrapped_loss_func = partial(loss_func, inputs)
    return output_tensor, wrapped_loss_func

class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        # pylint: disable=unused-argument
        # NOTE: force fp32
        vocab_parallel_logits = logits.float() if logits.dtype != torch.float32 else logits.clone()

        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())

        vocab_parallel_logits -= logits_max.unsqueeze(-1)
        exp_logits = vocab_parallel_logits.exp_()
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits, group=mpu.get_tensor_model_parallel_group())

        softmax_times_logits = (
            exp_logits
            .div_(sum_exp_logits.unsqueeze(-1))
            .mul_(logits)
            .sum(dim=-1)
        )
        dist.all_reduce(softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        return logits_max + sum_exp_logits.log() - softmax_times_logits

def entropy_from_tensor_parallel_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy loss on tp-sharded logits.

    Args:
        logits: (*, vocab_size // tp_size)
    """
    return _VocabParallelEntropy.apply(logits)

def get_ltor_masks_and_position_ids(
    data, eod_token, reset_position_ids, reset_attention_mask, eod_mask_loss, prompt_position_ids
):
    """
    Build masks and position id for left to right model.
    Copy from megatron.training.utils.get_ltor_masks_and_position_ids
    Patch for extra input of position_ids
    """

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
    ).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # ==================== VL Model Patch ====================
    if prompt_position_ids is not None:
        # for vl model we compute position ids in data_iter
        position_ids = torch.zeros((3, micro_batch_size, seq_length), dtype=torch.long, device=data.device)
        for batch_idx, batch in enumerate(prompt_position_ids):
            for list_idx in range(3):
                sublist = batch[list_idx]
                # Fill the beginning of the result tensor with the existing sublist
                position_ids[list_idx, batch_idx, :len(sublist)] = torch.tensor(sublist)
                # Determine the starting value for padding
                start_value = max(sublist) + 1
                # Fill the rest with increasing values starting from max(sublist) + 1
                position_ids[list_idx, batch_idx, len(sublist):] = torch.arange(
                    start_value, start_value + (seq_length - len(sublist))
                )
    # ==================== VL Model Patch ====================
    else:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
        position_ids = position_ids.unsqueeze(0).expand_as(data)
        # We need to clone as the ids will be modifed based on batch index.
        if reset_position_ids:
            position_ids = position_ids.clone()

        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(micro_batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, data[b] == eod_token]
                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.clone()

                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    # Mask attention loss.
                    if reset_attention_mask:
                        attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                    # Reset positions.
                    if reset_position_ids:
                        position_ids[b, (i + 1) :] -= i + 1 - prev_index
                        prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids
