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
"""constant"""

from enum import Enum
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .utils import get_global_statistics


def pad_to_max_len(all_tokens_right_padded, max_len, pad_value):
    pad_length = max_len - all_tokens_right_padded.size(1)

    # Pad the tensor with zeros on the right side to the desired length
    padded_tensor = torch.nn.functional.pad(all_tokens_right_padded, (0, pad_length), mode='constant', value=pad_value)
    return padded_tensor


def select_actions_from_right_padded(ts, action_starts, response_size, pad_value, dim):
    '''
    token to pad for different size of responses
    dim = dimension of ts to pad. e.g for logits.
    :param ts:
    :param action_starts:
    :param response_size:
    :return:
    '''
    res = []
    for t, action_start in zip(ts, action_starts):
        t_res = t[action_start:action_start + response_size]
        curr_len = t_res.size(dim)
        if dim == -1:
            pad_method = (0, response_size - curr_len)
        elif dim == -2:
            pad_method = (0, 0, 0, response_size - curr_len)
        else:
            raise NotImplementedError()
        t_res_padded = F.pad(input=t_res, pad=pad_method, mode="constant", value=pad_value)
        assert t_res_padded.size(dim=dim) == response_size
        res.append(t_res_padded)

    return torch.stack(res, dim=0).to(torch.cuda.current_device())


def get_ltor_masks_and_position_ids_rlhf(data):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    _, seq_length = data.size()

    att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, position_ids


class RunningMoments:
    """RunningMoments"""

    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def reset(self):
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """Updates running moments from batch's moments computed across ranks"""
        if dist.is_initialized():
            xs_mean, xs_var, xs_count = get_global_statistics(xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta ** 2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).sqrt()
        self.count = tot_count

        return xs_mean, (xs_var * xs_count / (xs_count - 1)).sqrt()


def get_running_stats(running_dict: Dict[str, RunningMoments]):
    res = {}
    for key, running in running_dict.items():
        res[f"{key}_mean"] = running.mean
        res[f"{key}_var"] = running.var
    return res


def reset_running_stats(running_dict: Dict[str, RunningMoments]):
    for _, running in running_dict.items():
        running.reset()


class TrainerEngine(str, Enum):
    """trainer engine.
        1. dpo: reference, policy_trainer
        2. online_dpo: policy, reward, reference, policy_trainer
        3. rlhf: policy, value, reward, reference, policy_trainer, value_trainer
    """
    DPO = "dpo"
    RLHF = "rlhf"
    ONLINE_DPO = "online_dpo"
    GRPO = "grpo"
