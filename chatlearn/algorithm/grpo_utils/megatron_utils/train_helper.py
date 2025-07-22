# pylint: skip-file
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

import copy
from typing import Dict
from functools import partial

import torch
from torch import distributed as dist

from megatron.core import mpu
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import (get_args, get_timers, is_last_rank, get_tokenizer,
                               print_rank_last)
from megatron.training.training import print_datetime
from megatron.training.utils import (average_losses_across_data_parallel_group,
                                     get_ltor_masks_and_position_ids,
                                     report_memory, unwrap_model)

from chatlearn.utils import to_device


def pad_to_max_len(all_tokens_right_padded, max_len, pad_value):
    pad_length = max_len - all_tokens_right_padded.size(1)
    if pad_length <= 0:
        return all_tokens_right_padded
    # Pad the tensor with zeros on the right side to the desired length
    padded_tensor = torch.nn.functional.pad(
        all_tokens_right_padded, (0, pad_length), mode="constant", value=pad_value
    )
    return padded_tensor


def generate_loss_mask_position_ids(
    tokens: torch.Tensor, prompt_token_length: list, response_token_length: list
):
    # Setup attention mask by prompt token length and response token length
    loss_mask = torch.zeros_like(tokens, dtype=torch.int32, device=tokens.device)
    for i in range(len(prompt_token_length)):
        loss_mask[
            i,
            prompt_token_length[i] : prompt_token_length[i] + response_token_length[i],
        ] = 1.0
    _, seq_len = tokens.size()
    position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0).expand_as(tokens)

    return loss_mask, position_ids


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
                total_loss_dict.get(key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
            )
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
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
    if (iteration % args.tensorboard_log_interval == 0) and is_last_rank():

        for key in loss_dict:
            iter_dict[f"{name}/{key}"] = loss_dict[key]
            consumed_train_samples_dict[f"{name}/" + key + " vs samples"] = loss_dict[
                key
            ]

        if grad_norm is not None:
            iter_dict[f"{name}/" + "grad_norm"] = grad_norm
            consumed_train_samples_dict[f"{name}/" + "grad-norm vs samples"] = grad_norm

        if more_grad_norm is not None:
            for k in more_grad_norm:
                iter_dict[f"{name}/{k}" + " grad_norm"] = more_grad_norm[k]
                consumed_train_samples_dict[f"{name}/{k}" + " grad-norm vs samples"] = (
                    more_grad_norm[k]
                )

        if params_norm is not None:
            iter_dict[f"{name}/" + "params-norm"] = params_norm
            consumed_train_samples_dict[f"{name}/" + "params-norm vs samples"] = (
                params_norm
            )

    if iteration % args.log_interval == 0:
        elapsed_time = 0
        elapsed_time_per_iteration = elapsed_time / total_iterations
        if args.log_timers_to_tensorboard:
            iter_dict[f"{name}/" + "iteration-time"] = elapsed_time_per_iteration

        log_string = " iteration {:8d}/{:8d} |".format(iteration, args.train_iters)
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
                if avg > 0.0:
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


def get_batch(batch_data):
    """Generate a batch"""
    args = get_args()

    data_b = next(batch_data)
    prompt_token_length = to_device("cuda", data_b["prompt_token_length"])
    response_token_length = to_device("cuda", data_b["response_token_length"])
    ref_logprobs = data_b["ref_logprobs"].float()
    old_logprobs = data_b["old_logprobs"].float()
    advantages = data_b["advantages"]
    tokens_ = data_b["all_tokens"].long()

    max_size = args.seq_length + 1
    tokens_ = pad_to_max_len(tokens_, max_size, pad_value=get_tokenizer().eod)

    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    loss_mask, _ = generate_loss_mask_position_ids(
        tokens, prompt_token_length, response_token_length
    )
    loss_mask = loss_mask[:, 1:]
    loss_mask = pad_to_max_len(loss_mask, args.seq_length, pad_value=0)

    # Get the masks and position ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        get_tokenizer().eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    inputs = {
        "all_tokens": tokens,
        "all_token_attention_mask": attention_mask,
        "all_token_position_ids": position_ids,
        "all_token_loss_mask": loss_mask,
        "labels": labels,
        "advantages": advantages,
        "ref_logprobs": ref_logprobs,
        "old_logprobs": old_logprobs,
    }

    for k, v in inputs.items():
        inputs[k] = to_device("cuda", v)
    return inputs


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
        final_loss = (loss.float() * loss_mask).sum()
        if key in require_bp_keys:
            total_loss_for_bp = total_loss_for_bp + final_loss

        reporting_losses[key] = final_loss.detach().clone().to(torch.float)

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_losses["num_tokens"] = num_tokens
    return total_loss_for_bp, num_tokens, reporting_losses


def forward_step(data_iterator, model):
    """Forward step."""

    inputs = get_batch(data_iterator)

    output_tensor = model(
        input_ids=inputs["all_tokens"],
        position_ids=inputs["all_token_position_ids"],
        attention_mask=inputs["all_token_attention_mask"],
        labels=inputs["labels"],
        training_inputs=inputs,
    )

    return output_tensor, partial(loss_func, inputs)


def inference_get_batch(data_iter):
    """Generate a batch"""
    args = get_args()
    data = next(data_iter)
    tokens_ = data["all_tokens"].long()

    # pad to max seq length or to tp*N
    max_size = args.seq_length + 1
    pad_all_tokens = pad_to_max_len(tokens_, max_size, pad_value=get_tokenizer().eod)

    labels = pad_all_tokens[:, 1:]
    tokens_ = pad_all_tokens[:, :-1]

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens_,
        get_tokenizer().eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    inputs = {
        "all_tokens": tokens_,
        "all_token_attention_mask": attention_mask,
        "all_token_position_ids": position_ids,
        "labels": labels,
    }

    for k, v in inputs.items():
        inputs[k] = to_device("cuda", v)
    return inputs


def inference_forward_step(data_iterator, model):
    """Forward step."""

    inputs = inference_get_batch(data_iterator)

    output_tensor = model(
        input_ids=inputs["all_tokens"],
        position_ids=inputs["all_token_position_ids"],
        attention_mask=inputs["all_token_attention_mask"],
        labels=inputs["labels"],
    )

    # NOTE: The latter (loss function) just returns the output tensor (the first argument).
    return output_tensor, lambda x, **_: x


class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
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