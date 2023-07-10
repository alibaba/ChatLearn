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
"""RLHF Policy Gradient Loss"""

import torch
from megatron.core import mpu
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.global_vars import get_args
from megatron.utils import average_losses_across_data_parallel_group


# pylint: disable=arguments-differ,abstract-method
class PolicyGradientLoss(torch.autograd.Function):
    """
    Policy Gradient Loss
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits, cliprange, action_ids, old_logprobs, advantages, loss_mask, stats):
        # Maximum value along vocab dimension across all GPUs.
        vocab_parallel_logits = vocab_parallel_logits.clone()  # for view error
        action_ids = action_ids.clone()
        old_logprobs = old_logprobs.clone()
        advantages = advantages.clone()
        loss_mask = loss_mask.clone()
        if get_args().numerical_stable:
            logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
            torch.distributed.all_reduce(logits_max,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=mpu.get_tensor_model_parallel_group())
            logits_min = torch.min(vocab_parallel_logits, dim=-1)[0]
            torch.distributed.all_reduce(logits_min,
                                         op=torch.distributed.ReduceOp.MIN,
                                         group=mpu.get_tensor_model_parallel_group())
            logits_max = (logits_min + logits_max) / 2

        else:
            logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
            torch.distributed.all_reduce(logits_max,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=mpu.get_tensor_model_parallel_group())
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = mpu.get_tensor_model_parallel_rank()
        world_size = mpu.get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (action_ids < vocab_start_index) | (
            action_ids >= vocab_end_index)  # [b,s] 1 for not in range action, 0 for in range
        # print(f"target_mask: {target_mask}")

        masked_actionids = action_ids.clone() - vocab_start_index  # [b,s]
        masked_actionids[target_mask] = 0  # [b,s]


        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)  # [n vp]
        masked_actionids_1d = masked_actionids.view(-1)  # [n] 0 for not in vocab range, target id -start for in range
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
        predicted_logits_1d = logits_2d[
            arange_1d, masked_actionids_1d]  # [n] in range target logit, not in range logits[0]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        action_logits = predicted_logits_1d.view_as(action_ids)
        action_logits[target_mask] = 0.0  # [b s] 0 for not in range, logit for in range
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(action_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_tensor_model_parallel_group())
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits  # [ b, s, vp ]
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.clone().sum(dim=-1)  # [ b, s ]
        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_tensor_model_parallel_group())

        action_logprob = action_logits - torch.log(sum_exp_logits + 1e-10)  # log ( exp(l) / sum(exp(li)
        # Loss = log(sum(exp(logits))) - predicted-logit.
        assert not torch.isnan(action_logprob).any(), f"action_logprob {action_logprob}"
        # Store softmax, target-mask and masked-target for backward pass.
        # exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        # clamp the diff to be exponentiated to be bounded

        if get_args().numerical_stable:
            logprob_diff = torch.clamp(action_logprob - old_logprobs, min=-1e5, max=1e5)
            log_ratio = (logprob_diff) * loss_mask
            # numerical approximate an exponential for numerical stability
            ratio = 1 + log_ratio + torch.square(log_ratio) / 2
        else:
            logprob_diff = action_logprob - old_logprobs
            log_ratio = (logprob_diff) * loss_mask
            ratio = torch.exp(log_ratio)
            # numerical approximate an exponential for numerical stability
            # ratio = 1 + log_ratio + torch.square(log_ratio) / 2
        # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
        assert not torch.isnan(ratio).any(), f"ratio {ratio} old_logprobs {old_logprobs}"

        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1) - log_ratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - cliprange,
            1.0 + cliprange,
        )

        loss = torch.max(pg_loss1, pg_loss2) * loss_mask  # [b, s]
        # loss = pg_loss1  # [b, s]
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * loss_mask).detach()
        torch.distributed.all_reduce(approx_kl,
                                     op=torch.distributed.ReduceOp.AVG,
                                     group=mpu.get_tensor_model_parallel_group())
        torch.distributed.all_reduce(pg_clipfrac,
                                     op=torch.distributed.ReduceOp.AVG,
                                     group=mpu.get_tensor_model_parallel_group())
        all_average_approx_kl, all_average_pg_clipfrac = average_losses_across_data_parallel_group(
            [approx_kl, pg_clipfrac])
        stats["policy/approx_kl"] = all_average_approx_kl.item()
        stats["policy/pg_clipfrac"] = all_average_pg_clipfrac.item()

        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))  # [b, s, v]
        selected_action_softmax_1d = torch.exp(action_logprob).view(-1)  # [n]

        ctx.save_for_backward(exp_logits, masked_actionids_1d, selected_action_softmax_1d, old_logprobs, advantages,
                              ratio, loss_mask, target_mask)
        ctx.cliprange = cliprange

        return loss  # [b, response size]

    @staticmethod
    def backward(ctx, grad_output):  # [b, resposne size]
        print(f"grad_output size: {grad_output.size()}")

        # masked_actionids_1d: [n]
        S, masked_actionids_1d, selected_action_softmax_1d, old_logprobs, advantages, ratio, mask, invalid_mask = ctx.saved_tensors

        cliprange = ctx.cliprange
        ratio_1d = ratio.view(-1)
        mask_1d = mask.view(-1)
        old_logprobs_1d = old_logprobs.view(-1)
        advantages_1d = advantages.view(-1)
        vocab = S.size(-1)
        s_2d = S.view(-1, vocab)  # [b*s, v]
        n = s_2d.size(0)
        invalid_mask = invalid_mask.view(-1)
        Sc = s_2d.clone()
        m = s_2d.clone()

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=n)  # [n]
        m[arange_1d, masked_actionids_1d] = invalid_mask * m[arange_1d, masked_actionids_1d]
        # [n] invalid actions [s1, s2, ... sn], valid actions [s1 ... 0, ... sn]
        Sc = Sc - m  # invalid action: [n, vp]
        grad_input = Sc - selected_action_softmax_1d.unsqueeze(-1) * s_2d  # [n, vp]
        grad_input *= -advantages_1d.unsqueeze(-1) * torch.exp(-old_logprobs_1d).unsqueeze(-1) * mask_1d.unsqueeze(-1)
        # Finally elementwise multiplication with the output gradients.
        # select clamped loss
        # grad = 0 : ratio < 1 - cliprange * advantages < 0 + ratio > 1 + cliprange * advantages > 0

        grad_input[(ratio_1d < 1.0 - cliprange) * (advantages_1d < 0)] = 0
        grad_input[(ratio_1d > 1.0 + cliprange) * (advantages_1d > 0)] = 0
        grad_input = grad_input.view_as(S)

        # add entropy gradient here:

        grad_input.mul_(grad_output.unsqueeze(-1))

        return grad_input.contiguous(), None, None, None, None, None, None


def tensor_decomp_pg_loss(config, action_token_logits, action_ids,
                          action_loss_mask, old_logprobs, advantages, stats):
    """Helper function for the cross entropy."""

    assert action_token_logits.size(1) == action_ids.size(1) \
           == action_loss_mask.size(1) == old_logprobs.size(1) == advantages.size(
        1), f"{action_token_logits.size(1)}, {action_ids.size(1)}," \
            f"{action_loss_mask.size(1)}, {old_logprobs.size(1)}," \
            f"{advantages.size(1)}"

    return PolicyGradientLoss.apply(action_token_logits, config.cliprange, action_ids,
                                     old_logprobs, advantages, action_loss_mask, stats)  # [b, response_size]

# pylint: enable=arguments-differ,abstract-method
