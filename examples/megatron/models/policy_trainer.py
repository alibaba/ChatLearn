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
"""policy trainer"""

from functools import partial

import numpy as np
import torch
from megatron import get_num_microbatches
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.core import mpu
from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import calc_params_l2_norm
from models.policy_model import PolicyModel
from models.utils import training_log

from chatlearn.utils import to_device
from .base_trainer import BaseTrainer
from .constants_ppo import select_actions_from_right_padded, get_ltor_masks_and_position_ids, pad_to_max_len
from .utils import get_eos_id


class AdaptiveKLController:
    """Adaptive KL Controller as described in Ziegler et al. "Fine-Tuning Language Models from Human Preferences"
    Reference: Section 2.2 https://arxiv.org/pdf/1909.08593.pdf#page=2
    Source: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py
    """

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int):
        """Returns adaptively updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)  # ϵₜ
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # βₜ₊₁


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current: float, n_steps: int):
        """Returns updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """

class PolicyTrainer(BaseTrainer):
    """gpt model wrapper"""

    def setup(self):
        super().setup()
        self.kl_ctl = AdaptiveKLController(
            self.args.init_kl_coef, self.args.target, self.args.horizon
        )

    def model_provider(self, pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building GPT model ...')
        model = PolicyModel(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            stats=self.stats
        )
        if self.module_args.lora.enable_lora:
            from chatlearn.models.megatron.lora import convert_layer_to_lora # pylint: disable=import-outside-toplevel
            model = convert_layer_to_lora(model)
        return model

    def get_batch(self, batch_data):
        """Generate a batch
            "all_token_ids_right_padded": torch.tensor([[p,p,5,6,7], [p,p,p,8,9]], dtype=torch.long, device=device),
            "action_start_indices": torch.tensor([[10,100,p,p,p], [11,p,p,p,p]], dtype=torch.long, device=device),
            "action_logprobs": torch.randn([bs, 5], dtype=torch.float32, device=device),
            "action_values": torch.randn([bs, 5], dtype=torch.float32, device=device),
            "action_rewards": torch.randn([bs, 5], dtype=torch.float32, device=device),
            "loss_mask"
        """
        args = self.args
        data_b = next(batch_data)

        # TODO: move to RLHF framework later. add pad to max length config
        all_token_ids_right_padded = pad_to_max_len(data_b["all_token_ids_right_padded"], args.seq_length,
                                                    pad_value=get_eos_id(get_tokenizer()))
        all_token_loss_mask = pad_to_max_len(data_b["loss_mask"], args.seq_length, pad_value=0)

        all_token_attention_mask, all_token_position_ids = get_ltor_masks_and_position_ids(
            all_token_ids_right_padded)

        inputs = {
            "all_token_position_ids": all_token_position_ids,
            "all_token_ids_right_padded": all_token_ids_right_padded,
            # this attention mask is not TRANSFOEMRER attention msak. this actually applies on attention result [b, np, s, s]
            "all_token_attention_mask": all_token_attention_mask.bool(),
            "all_token_loss_mask": all_token_loss_mask.bool(),
            "action_starts": data_b['action_start_indices'],
            "action_logprobs": data_b["action_logprobs"].float(),  # response size
            "action_values": data_b["action_values"].float(),
            "action_rewards": data_b["action_rewards"].float(),
        }

        assert all_token_ids_right_padded.size(0) != 1, f"cannot be 1 will be squeezed. {all_token_ids_right_padded}"

        for k, v in inputs.items():
            inputs[k] = to_device("cuda", v)
        return inputs

    def aggregate_loss_func(self, inputs, losses):  # [b, s]
        # losses = losses.float()
        # b = losses.size(0)
        # loss = torch.sum(losses.view(-1)) / b

        losses = losses.float()  # [b, response_size]

        old_rewards = inputs['action_rewards']  # [b, responses size]
        response_length = old_rewards.shape[1]
        # Note the tkoken logits to get loss is only the actions. query doesn't have loss.
        action_loss_mask = select_actions_from_right_padded(ts=inputs["all_token_loss_mask"],
                                                            action_starts=inputs["action_starts"] - 1,
                                                            # because align iwth logits index
                                                            response_size=response_length,
                                                            pad_value=0,
                                                            dim=-1).contiguous()

        action_loss_mask = action_loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * action_loss_mask) / action_loss_mask.sum()

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])
        # Reduce loss for logging.
        self.stats["policy_loss"] = averaged_loss[0]
        return loss, {'policy lm loss': averaged_loss[0]}

    def _forward_step(self, batch_data, model):
        """Forward step."""
        inputs = self.get_batch(batch_data)
        losses = model.forward(all_token_ids=inputs["all_token_ids_right_padded"],
                               all_position_ids=inputs["all_token_position_ids"],
                               all_token_attention_mask=inputs["all_token_attention_mask"],
                               training_inputs=inputs)

        return losses, partial(self.aggregate_loss_func,
                               inputs)  # will call loss_func(loss_mask, output_tensor) to get loss


    def post_update_stuffs(self, loss_dict, skipped_iter,
                           grad_norm, num_zeros_in_grad, iteration):

        # only last rank give kl coef.
        if torch.distributed.get_rank() == (
            torch.distributed.get_world_size() - 1):
            self.kl_ctl.update(self.stats["policy/approx_kl"], n_steps=self.args.global_batch_size)
            if not self.args.fix_kl_coef:
                self.put("kl_coef", self.kl_ctl.value)

        # TODO: get_num_microbatches scheduler is constants so it's fine for now. but if not we need 2 args
        self.args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
                                            self.args.micro_batch_size * \
                                            get_num_microbatches()

        # Logging.
        loss_scale = self.optimizer.get_loss_scale().item()
        params_norm = None
        if self.args.log_params_norm:
            params_norm = calc_params_l2_norm(self.model)
        if self.args.log_interval > 0 and iteration % self.args.log_interval == 0:
            training_log(loss_dict, {},
                         self.optimizer.param_groups[0]['lr'],
                         iteration, loss_scale, skipped_iter,
                         grad_norm, params_norm, num_zeros_in_grad, self.stats,
                         name="policy_trainer")

# pylint: enable=unused-variable,invalid-name
