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
"""value trainer"""

from functools import partial

import torch
from megatron.core import mpu
try:
    from megatron.training import get_num_microbatches
except ImportError:
    from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.training import print_rank_0
from megatron.training.global_vars import get_tensorboard_writer
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.utils import calc_params_l2_norm

from chatlearn.utils import to_device
from .value_model import ValueModel as LegacyValueModel
from .mcore_value_model import MCoreValueModel
from .utils import tensorboard_scalar_dict, training_log, get_eos_id
from .base_trainer import BaseTrainer
from .constants import get_ltor_masks_and_position_ids_rlhf, select_actions_from_right_padded, pad_to_max_len


class ValueTrainer(BaseTrainer):
    """gpt model wrapper"""

    def model_provider(self, pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building GPT model ...')
        if self.args.use_legacy_models:
            model = LegacyValueModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                stats=self.stats,
                buffer=self.buffer
            )
            if self.module_args.lora.enable_lora:
                from chatlearn.models.megatron.lora import convert_layer_to_lora # pylint: disable=import-outside-toplevel
                model = convert_layer_to_lora(model)
        else:
            model = MCoreValueModel(
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                stats=self.stats,
                buffer=self.buffer
            )
            if self.module_args.lora.enable_lora:
                assert False, "ChatLearn do not support LoRA with Megatron-Core models currently."

        return model

    def get_batch(self, batch_data):
        """Generate a batch
            "all_token_ids_right_padded": torch.tensor([[p,p,5,6,7], [p,p,p,8,9]], dtype=torch.long, device=device),
            "action_start_indices": torch.tensor([[10,100,p,p,p], [11,p,p,p,p]], dtype=torch.long, device=device),
            "action_logprobs": torch.randn([bs, 5], dtype=torch.float32, device=device),
            "action_values": torch.randn([bs, 5], dtype=torch.float32, device=device),
            "action_rewards": torch.randn([bs, 5], dtype=torch.float32, device=device),
        """
        args = self.args
        data_b = next(batch_data)

        # TODO: move to ChatLearn framework later. add pad to max length config
        all_token_ids_right_padded = pad_to_max_len(data_b["all_token_ids_right_padded"], args.seq_length,
                                                    pad_value=get_eos_id(get_tokenizer()))
        # NOTE this pad to max is even better than get_loss_mask again because for the maxedout response cases,
        # get_loss_mask will add a loss mask = 1 to the first eod token which is WRONG because they didn't want
        # to stop and most likely it shouldn't stop. it's just maxed out.
        all_token_loss_mask = pad_to_max_len(data_b["loss_mask"], args.seq_length, pad_value=0)

        all_token_attention_mask, all_token_position_ids = get_ltor_masks_and_position_ids_rlhf(
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
        for k, v in inputs.items():
            inputs[k] = to_device("cuda", v)

        return inputs

    def aggregate_loss_func(self, inputs, losses):  # [b, s]

        losses = losses.float()  # [b, response_size]

        old_rewards = inputs['action_rewards']  # [b, responses size]
        response_length = old_rewards.shape[1]
        # we want to mask logits which is the previous tokens of an action!!! so -1
        action_loss_mask = select_actions_from_right_padded(ts=inputs["all_token_loss_mask"],
                                                            action_starts=inputs["action_starts"] - 1,
                                                            # because align iwth logits index
                                                            response_size=response_length,
                                                            pad_value=0, dim=-1).contiguous()
        action_loss_mask = action_loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * action_loss_mask) / action_loss_mask.sum()

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        # Reduce loss for logging.
        stats_update = {"value_loss": averaged_loss[0]}
        self.stats.update(stats_update)
        return loss, {'lm loss': averaged_loss[0]}

    def _forward_step(self, batch_data, model):
        """Forward step."""
        # args = get_args()
        timers = get_timers()

        # Get the batch.
        timers('batch-generator').start()
        inputs = self.get_batch(
            batch_data)
        timers('batch-generator').stop()

        losses = model.forward(all_token_ids=inputs["all_token_ids_right_padded"],
                               all_position_ids=inputs["all_token_position_ids"],
                               all_token_attention_mask=inputs["all_token_attention_mask"],
                               training_inputs=inputs)

        return losses, partial(self.aggregate_loss_func,
                               inputs)  # will call loss_func(loss_mask, output_tensor) to get loss


    def after_episode(self):
        '''
        ChatLearn calling
        :return:
        '''
        if self.args.log_interval <= 0:
            return
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                torch.distributed.get_world_size() - 1):
                gathered_returns = torch.cat(self.buffer["value/returns"], dim=0).view(-1)  # [b* dp, max_response_size]

                gathered_value_preds = torch.cat(self.buffer["value/value_preds"], dim=0).view(-1)  # same

                return_not_zero_mask = gathered_returns.nonzero()
                gathered_returns = gathered_returns[return_not_zero_mask]
                gathered_value_preds = gathered_value_preds[return_not_zero_mask]

                # Create a ground truth tensor and a predicted tensor
                y_true = gathered_returns
                y_pred = gathered_value_preds

                # Calculate the mean and variance of the error
                var_y = torch.var(y_true)
                explained_var = torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y
                # Print the explained variance
                print("Explained variance:", explained_var)
                self.stats["value/explained_variance_dp"] = explained_var

                # actual log
                writer = get_tensorboard_writer()

                after_episode_dict = {
                    "value/explained_variance_dp": self.stats["value/explained_variance_dp"]
                }
                tensorboard_scalar_dict(writer, prefix="", global_step=self.args.consumed_train_samples,
                                        scalar_dict=after_episode_dict)

    def before_episode(self):
        '''
        ChatLearn calling
        :return:
        '''
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                torch.distributed.get_world_size() - 1):
                self.buffer["value/returns"] = []
                self.buffer["value/value_preds"] = []

    def post_update_stuffs(self, loss_dict, skipped_iter,
                           grad_norm, num_zeros_in_grad, iteration):

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
                         name="value_trainer")

# pylint: enable=unused-variable,invalid-name
