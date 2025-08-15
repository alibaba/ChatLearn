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

from flash_attn.bert_padding import pad_input, unpad_input

from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import (
    get_args, get_timers,
    is_last_rank,
    get_tokenizer,
    print_rank_last
)
from megatron.training.training import print_datetime
from megatron.training.utils import (
    get_ltor_masks_and_position_ids,
    report_memory,
    unwrap_model
)

try:
    import transformer_engine  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.utils import is_te_min_version

    HAVE_TE = True
    try:
        import transformer_engine_torch as tex

        HAVE_TEX = True
    except:
        HAVE_TEX = False
except:
    HAVE_TE = False

from chatlearn.utils import to_device



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
                total_loss_dict.get(key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
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

    elapsed_time = 0
    elapsed_time_per_iteration = elapsed_time / total_iterations
    if args.log_timers_to_tensorboard:
        iter_dict[f"{name}/" + "iteration-time"] = elapsed_time_per_iteration

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

    all_tokens = data_b["all_tokens"].long().cuda() # shape: [mbs, seq_length]
    mbs, ori_seqlen = all_tokens.shape
    prompt_token_length = to_device("cuda", data_b["prompt_token_length"])
    response_token_length = to_device("cuda", data_b["response_token_length"])

    if is_packing:
        attn_mask_for_unpadding = torch.zeros_like(all_tokens, dtype=torch.int32)
        for i, (prompt_length, response_length) in enumerate(
            zip(prompt_token_length, response_token_length)
        ):
            attn_mask_for_unpadding[i, : prompt_length + response_length] = 1

        if args.sequence_parallel and mpu.get_context_parallel_world_size() == 1:
            # all_tokens: [9, 2048], attn_mask_for_unpadding: [9, 2048], unpad_tokens: [7851, 1], indices: [7851]
            # cu_seqlens: [10]->tensor([   0, 2048, 3761, 4801, 5610, 6230, 6773, 7233, 7591, 7851]
            # max_seqlen_in_batch: 2048
            (
                unpad_tokens,
                indices,
                cu_seqlens,
                max_seqlen_in_batch,
                *_
            ) = unpad_input(all_tokens.unsqueeze(-1), attn_mask_for_unpadding)

            unpad_tokens = unpad_tokens.transpose(0, 1) # [s, b] -> [b, s]
            # NOTE: sequence-parallel padding, the len of right-padded seq should be divisible by TP & Expert TP
            pad_size = 0
            divisor = mpu.get_tensor_model_parallel_world_size()
            total_nnz = unpad_tokens.shape[1]
            pad_size = (divisor - total_nnz % divisor) % divisor
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size), value=get_tokenizer().eod)
            max_seqlen_in_batch = max(max_seqlen_in_batch, pad_size)
            cu_seqlens = torch.cat([
                cu_seqlens,
                torch.tensor(unpad_tokens.shape[1], dtype=cu_seqlens.dtype, device=cu_seqlens.device).unsqueeze(-1)
            ])

            labels = torch.roll(unpad_tokens, shifts=-1, dims=1)
            position_ids = torch.arange(unpad_tokens.shape[1], device=unpad_tokens.device).view(1, -1)
            prev_cu_seqlen = 0
            for cu_seqlen in cu_seqlens[1:]:
                position_ids[:, cu_seqlen:] -= cu_seqlen - prev_cu_seqlen
                prev_cu_seqlen = cu_seqlen

            packed_seq_params = PackedSeqParams(
                qkv_format='thd',
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max_seqlen_in_batch,
                max_seqlen_kv=max_seqlen_in_batch
            )

            input_data = {
                "all_tokens": unpad_tokens,
                "all_token_attention_mask": None,
                "all_token_position_ids": position_ids,
                "labels": labels,
                'packed_seq_params': packed_seq_params,
                "ori_seq_len": ori_seqlen,
                "ori_batch_size": mbs,
                "indices": indices,
                "pad_size": pad_size,
            }
        elif args.sequence_parallel and mpu.get_context_parallel_world_size() >1:
            tp_size = mpu.get_tensor_model_parallel_world_size()
            cp_size = mpu.get_context_parallel_world_size()
            cp_rank = mpu.get_context_parallel_group().rank()
            align_size = tp_size * cp_size* 2
            seqlens_in_batch = attn_mask_for_unpadding.sum(dim=-1, dtype=torch.int32)
            pad_size = (align_size - seqlens_in_batch % align_size) % align_size
            seqlens_in_batch_padded = seqlens_in_batch + pad_size
            cu_seqlens = torch.zeros(mbs + 1, dtype=torch.int32, device=all_tokens.device)
            cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
            cu_seqlens_padded = torch.zeros(mbs + 1, dtype=torch.int32, device=all_tokens.device)
            cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)
            max_seqlen_in_batch = seqlens_in_batch_padded.max().item()
            shape = list(all_tokens.shape[1:])
            shape[0] = seqlens_in_batch_padded.sum().item() // cp_size
            input_ids_rmpad = torch.zeros(shape, dtype=all_tokens.dtype, device=all_tokens.device)
            labels_rmpad = torch.zeros(shape, dtype=all_tokens.dtype, device=all_tokens.device)
            attn_mask_rmpad = torch.zeros(shape, dtype=all_tokens.dtype, device=all_tokens.device)
            attn_mask_for_unpadding = (attn_mask_for_unpadding > 0.5)
            chunked_seqlens_in_batch_padded = []
            chunked_pad_size_in_batch = []
            for i in range(mbs):
                seqlen = seqlens_in_batch_padded[i] // cp_size
                half_seqlen = seqlen // 2
                start_idx = cu_seqlens_padded[i] // cp_size
                next_start_idx = cu_seqlens_padded[i+1] // cp_size
                d = all_tokens[i, attn_mask_for_unpadding[i]]
                # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
                # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1
                input_ids_rmpad[start_idx:start_idx + half_seqlen] = d[half_seqlen * cp_rank:half_seqlen * (cp_rank + 1)]
                attn_mask_rmpad[start_idx:start_idx + half_seqlen] = 1
                labels_rmpad = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
     
                # 1536
                remain_start = seqlens_in_batch_padded[i] - half_seqlen * (cp_rank + 1)
                # 2048
                remain_end = seqlens_in_batch_padded[i] - half_seqlen * cp_rank
                remain_end = min(remain_end, d.shape[0])
                remain_len = remain_end - remain_start
                
                if remain_len > 0:
                    input_ids_rmpad[start_idx + half_seqlen:start_idx + half_seqlen +
                                    remain_len] = d[remain_start:remain_end]
                    
                    attn_mask_rmpad[start_idx + half_seqlen:start_idx + half_seqlen +
                                    remain_len] = 1
                    local_pad_size = next_start_idx - start_idx - (half_seqlen + remain_len) 
                    chunked_seqlens_in_batch_padded.append(half_seqlen + remain_len + local_pad_size)
                    chunked_pad_size_in_batch.append(local_pad_size)
                else:
                    local_pad_size = next_start_idx - start_idx - half_seqlen
                    chunked_seqlens_in_batch_padded.append(half_seqlen + local_pad_size)
                    chunked_pad_size_in_batch.append(local_pad_size)

                labels_rmpad = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
                    
            chunked_seqlens_in_batch_padded = torch.stack(chunked_seqlens_in_batch_padded)
            chunked_pad_size_in_batch = torch.stack(chunked_pad_size_in_batch)
            chunked_cu_seqlens_padded = torch.zeros(mbs + 1, dtype=torch.int32, device=all_tokens.device)
            chunked_cu_seqlens_padded[1:] = torch.cumsum(chunked_seqlens_in_batch_padded, dim=0)

            packed_seq_params = PackedSeqParams(qkv_format='thd',
                                                cu_seqlens_q=cu_seqlens_padded,
                                                max_seqlen_q=max_seqlen_in_batch,
                                                cu_seqlens_kv=cu_seqlens_padded,
                                                max_seqlen_kv=max_seqlen_in_batch,
                                                cu_seqlens_q_padded=cu_seqlens_padded,
                                                cu_seqlens_kv_padded=cu_seqlens_padded)

            indices = torch.nonzero(attn_mask_rmpad.flatten(), as_tuple=False).flatten()

            input_ids_rmpad = input_ids_rmpad.unsqueeze(0)

            loss_mask = torch.ones(labels_rmpad.size(), dtype=torch.float, device=input_ids_rmpad.device)

            loss_mask[labels_rmpad == 0] = 0.0

            labels = labels_rmpad.unsqueeze(0)

            position_ids = torch.arange(input_ids_rmpad.shape[1], device=input_ids_rmpad.device).view(1, -1)
            prev_cu_seqlen_padded = 0
            for cu_seqlen_padded in chunked_cu_seqlens_padded[1:]:
                position_ids[:, cu_seqlen_padded:] -= cu_seqlen_padded - prev_cu_seqlen_padded
                prev_cu_seqlen_padded = cu_seqlen_padded

            chunked_max_seqlen_in_batch = chunked_seqlens_in_batch_padded.max().item()
            input_data = {
                "all_tokens": input_ids_rmpad,
                "all_token_attention_mask": attn_mask_rmpad,
                "all_token_loss_mask": loss_mask,
                "all_token_position_ids": position_ids,
                "labels": labels,
                'packed_seq_params': packed_seq_params,
                "ori_seq_len": ori_seqlen // cp_size,
                "ori_batch_size": mbs,
                "indices": indices,
                "pad_size": chunked_pad_size_in_batch,
            }


    else:
        labels = torch.roll(tokens, shifts=-1, dims=1)
        # Get the masks and position ids.
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            get_tokenizer().eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss,
        )

        input_data = {
            "all_tokens": tokens,
            "all_token_attention_mask": attention_mask,
            "all_token_position_ids": position_ids,
            "labels": labels,
            'packed_seq_params': None
        }

    if is_training:
        ref_logprobs = data_b["ref_logprobs"].float()
        old_logprobs = data_b["old_logprobs"].float()
        advantages = data_b["advantages"]

        if mpu.get_context_parallel_world_size() == 1:
            loss_mask = torch.zeros_like(data_b["all_tokens"], dtype=torch.int32)
            for i, (prompt_length, response_length) in enumerate(
                zip(prompt_token_length, response_token_length)
            ):
                loss_mask[i, prompt_length: prompt_length + response_length] = 1
            
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1)

        elif mpu.get_context_parallel_world_size() > 1:
            loss_mask = input_data['all_token_loss_mask']
            attention_mask = input_data["all_token_attention_mask"]> 0.5
            loss_mask = pad_input(
                loss_mask[attention_mask].unsqueeze(-1),
                input_data['indices'], 
                input_data['ori_batch_size'], 
                input_data['ori_seq_len']
            ).squeeze(-1)

            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1)

        input_data.update({
            "all_token_loss_mask": loss_mask,
            "advantages": advantages,
            "ref_logprobs": ref_logprobs,
            "old_logprobs": old_logprobs,
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

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_losses["num_tokens"] = num_tokens
    reporting_losses["num_samples"] = torch.ones_like(num_tokens)
    return total_loss_for_bp, num_tokens, reporting_losses


def forward_step(data_iterator, model, *, is_training: bool=False, is_packing: bool=False):
    """Forward step."""

    inputs = get_batch(
        data_iterator,
        is_training=is_training,
        is_packing=is_packing
    )
    # TODO: refactor model.forward to return logits or logprobs, and make
    # loss computation in loss_func
    output_tensor = model(
        input_ids=inputs["all_tokens"],
        position_ids=inputs["all_token_position_ids"],
        attention_mask=inputs["all_token_attention_mask"],
        labels=inputs["labels"],
        training_inputs=inputs if is_training else None,
        packed_seq_params=inputs['packed_seq_params'] if is_packing else None
    )

    if is_training:
        wrapped_loss_func = partial(loss_func, inputs)
    else:
        if unwrap_model(model).post_process and is_packing:
            if mpu.get_context_parallel_world_size() == 1:
                output_tensor = pad_input(
                    output_tensor[0, :output_tensor.shape[1] - inputs['pad_size']].unsqueeze(-1),
                    inputs['indices'],
                    inputs['ori_batch_size'],
                    inputs['ori_seq_len']
                ).squeeze(-1)

            elif mpu.get_context_parallel_world_size() > 1:
                attention_mask = inputs["all_token_attention_mask"]> 0.5
                output_tensor = pad_input(
                    output_tensor[0][attention_mask].unsqueeze(-1),
                    inputs['indices'],
                    inputs['ori_batch_size'],
                    inputs['ori_seq_len']
                ).squeeze(-1)

        # NOTE: just returns the output tensor (the first argument).
        wrapped_loss_func = lambda x, **_: x # pylint: disable=unnecessary-lambda-assignment

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
