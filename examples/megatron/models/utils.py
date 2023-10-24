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
"""utils"""

import json
import inspect
import math
import os
import random
from numbers import Number
from pathlib import Path
from typing import Tuple, Optional

import jsonlines
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron import print_rank_last, is_last_rank, get_num_microbatches, get_args, get_timers
from megatron.core import mpu
from megatron.global_vars import get_tensorboard_writer
from megatron.training import print_datetime
from torchtyping import TensorType


def logprobs_from_logits(logits, labels):
    """Compute log softmax values from logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)


def get_global_statistics(xs: torch.Tensor) -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    """
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    dist.all_reduce(sum_and_count, dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum((xs - global_mean) ** 2)
    dist.all_reduce(sum_var, dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
    global_var = sum_var / count
    return global_mean, global_var, count


def whiten(xs: torch.Tensor, shift_mean=True, distributed=True) -> torch.Tensor:
    """Whitens values"""
    if distributed and dist.is_initialized():
        mean, var, _ = get_global_statistics(xs)
    else:
        var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def get_advantages_and_returns(
    config,
    values: TensorType["batch_size", "response_size"],
    rewards: TensorType["batch_size", "response_size"],
    response_length: int,
    use_whitening: Optional[bool] = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lastgaelam = 0
    advantages_reversed = []
    for t in reversed(range(response_length)):
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        delta = rewards[:, t] + config.gamma * nextvalues - values[:, t]
        lastgaelam = delta + config.gamma * config.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    if use_whitening:
        advantages = whiten(advantages)

    # both [b, response_size] == size of value
    return advantages.detach(), returns


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad, stats, name):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf')
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'grads-all-reduce',
        'grads-reduce-scatter',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
                 get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    iter_dict = {}
    consumed_train_samples_dict = {}
    # Tensorboard values.
    if (iteration % args.tensorboard_log_interval == 0) and \
        is_last_rank():

        if args.log_learning_rate_to_tensorboard:
            iter_dict[f'{name}/learning-rate'] = learning_rate
            consumed_train_samples_dict[f'{name}/learning-rate vs samples'] = learning_rate

        if args.log_batch_size_to_tensorboard:
            iter_dict[f'{name}/batch'] = batch_size
            consumed_train_samples_dict[f'{name}/batch-size vs samples'] = batch_size

        for key in loss_dict:
            iter_dict[f'{name}/{key}'] = loss_dict[key]
            consumed_train_samples_dict[f'{name}/' + key + ' vs samples'] = loss_dict[key]

        if args.log_loss_scale_to_tensorboard:
            iter_dict[f'{name}/' + 'loss-scale'] = loss_scale
            consumed_train_samples_dict[f'{name}/' + 'loss-scale vs samples'] = loss_scale

        if args.log_world_size_to_tensorboard:
            iter_dict[f'{name}/' + 'world-size'] = args.world_size
            consumed_train_samples_dict[f'{name}/' + 'world-size vs samples'] = args.world_size

        if grad_norm is not None:
            iter_dict[f'{name}/' + 'grad-norm'] = grad_norm
            consumed_train_samples_dict[f'{name}/' + 'grad-norm vs samples'] = grad_norm

        if num_zeros_in_grad is not None:
            iter_dict[f'{name}/' + 'num-zeros'] = num_zeros_in_grad
            consumed_train_samples_dict[f'{name}/' + 'num-zeros vs samples'] = num_zeros_in_grad

        if params_norm is not None:
            iter_dict[f'{name}/' + 'params-norm'] = params_norm
            consumed_train_samples_dict[f'{name}/' + 'params-norm vs samples'] = params_norm

        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            iter_dict[f'{name}/' + "mem-reserved-bytes"] = mem_stats["reserved_bytes.all.current"]
            iter_dict[f'{name}/' + "mem-allocated-bytes"] = mem_stats["allocated_bytes.all.current"]
            iter_dict[f'{name}/' + "mem-allocated-count"] = mem_stats["allocation.all.current"]

    if iteration % args.log_interval == 0:
        # TODO: fixed later, we should call timers('interval-time', log_level=0).start(barrier=True) first in the begining
        # elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time = 0
        elapsed_time_per_iteration = elapsed_time / total_iterations
        if args.log_timers_to_tensorboard:
            iter_dict[f'{name}/' + 'iteration-time'] = elapsed_time_per_iteration

        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)

        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                    iter_dict[key] = avg

                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        iter_dict[f'{name}/' + "loss scale"] = loss_scale

        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)

        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)

        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)

        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        iter_dict[f'{name}/' + "number of skipped iterations"] = total_loss_dict[skipped_iters_key]

        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        print_datetime('Logger')
        timers.log(timers_to_log, normalizer=args.log_interval)

        # RL related stats: global
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                torch.distributed.get_world_size() - 1):
                # actual log
                tensorboard_scalar_dict(writer, prefix="", global_step=args.consumed_train_samples, scalar_dict=stats)
                tensorboard_scalar_dict(writer, prefix="", global_step=args.consumed_train_samples,
                                        scalar_dict=iter_dict)
                tensorboard_scalar_dict(writer, prefix="", global_step=args.consumed_train_samples,
                                        scalar_dict=consumed_train_samples_dict)


        else:
            # actual log
            tensorboard_scalar_dict(writer, prefix="", global_step=args.consumed_train_samples, scalar_dict=iter_dict)
            tensorboard_scalar_dict(writer, prefix="", global_step=args.consumed_train_samples,
                                    scalar_dict=consumed_train_samples_dict)
            tensorboard_scalar_dict(writer, prefix="", global_step=args.consumed_train_samples, scalar_dict=stats)


def get_tensor_stats(xs: torch.Tensor, mask: torch.Tensor, n: int):
    mean = (xs * mask).sum() / n
    return {'mean': mean, 'min': torch.where(mask.bool(), xs, np.inf).min(),
            'max': torch.where(mask.bool(), xs, -np.inf).max(),
            'std': torch.sqrt(((xs - mean) * mask).pow(2).sum() / n)}

def print_rank_0(*message):
    """
    Print only once from the main rank
    """
    if os.environ.get("RANK", "0") == "0":
        print(*message)


def significant(x: Number, ndigits=2) -> Number:
    """
    Cut the number up to its `ndigits` after the most significant
    """
    if isinstance(x, torch.Tensor):
        x = x.item()

    if not isinstance(x, Number) or x == 0:
        return x

    return round(x, ndigits - int(math.floor(math.log10(abs(x)))))


def set_seed(seed: int):
    """
    Sets seeds across package dependencies for reproducibility.
    """
    seed += int(os.environ.get("RANK", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def listdict_to_dictlist(ld, list_extend=True):
    '''
    [{k1: v11, k2: v2}, {k1: v12, k2: v2},....] => {k1: [v11, v12..], k2: [v21, v22...]}
    if v11 is list then k1: v11 + v12
    :param ld:
    :return:
    '''
    res = ld[0]
    for res_key, v in res.items():
        if list_extend and isinstance(res[res_key], list):
            continue

        res[res_key] = [v]

    for d in ld[1:]:
        for key, v in d.items():
            if list_extend and isinstance(d[key], list):
                res[key].extend(v)
            else:
                res[key].append(v)

    return res


def tensorboard_scalar_dict(tensorboard_writer, prefix, global_step, scalar_dict):
    if isinstance(scalar_dict, (float, int)):
        name = prefix
        value = scalar_dict
        tensorboard_writer.add_scalar(name, value, global_step)
    else:
        for key, value in scalar_dict.items():
            name = '{}/{}'.format(prefix, key)
            tensorboard_writer.add_scalar(name, value, global_step)


def get_loss_mask(all_tokens_right_padded, pad_token_id, prompt_sizes):
    '''
    if prompt_sizes is None means it doesn't care about if
    :param all_tokens_right_padded:
    :param pad_token_id:
    :param prompt_sizes:
    :return:
    '''
    loss_mask = (
        all_tokens_right_padded.not_equal(pad_token_id)
            .to(torch.cuda.current_device())
    )
    # we don't just caclulate loss on the action tokens but also the first pad token if present

    #
    # all_tokens_right_padded len = is the max length of the prompts + max generation tokens, so if there is no 0, take it as max len reached.
    # thus just
    occurrences = (all_tokens_right_padded == pad_token_id).float()

    row_indices = torch.arange(all_tokens_right_padded.size(1)).unsqueeze(0).expand(all_tokens_right_padded.size(0),
                                                                                    -1).to(
        all_tokens_right_padded.device)
    response_mask = (row_indices >= prompt_sizes.unsqueeze(1)).int()

    # mask out the stop appear in the prompt:
    occurrences = occurrences * response_mask

    first_stop_sequence_indices = torch.argmax(occurrences, dim=1)

    # if not found a stop sequence, occurrence will be sum 0 dim1
    not_found_mask = torch.sum(occurrences, dim=1) == 0
    # for the not found one. take stop_sequence = tokens.size(1)-1 before everything else replace tokens afterwards

    for i in range(loss_mask.size(0)):
        if not_found_mask[i] == 0:
            # if not not found = found a stop sequence.
            loss_mask[i, first_stop_sequence_indices[i]] = 1

    return loss_mask


def read_jsonl(file_path):
    print(f"read_jsonl from : {file_path}")

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, encoding="utf-8") as f1:
        res = [json.loads(line) for line in f1]

    return res


def write_jsonl(dict_list, fp):
    Path(fp).parent.mkdir(parents=True, exist_ok=True)
    print(f"writing to {fp}")
    with open(fp, 'w', encoding='utf-8') as f:
        writer = jsonlines.Writer(f)
        for item in dict_list:
            writer.write(item)

def contain_arg_name(func, arg_name):
    return arg_name in inspect.getfullargspec(func).args


def has_config_in_args(func):
    return contain_arg_name(func, 'config')


def get_eos_id(tokenizer):
    return tokenizer.eos_id if hasattr(tokenizer, 'eos_id') else tokenizer.eod_id
