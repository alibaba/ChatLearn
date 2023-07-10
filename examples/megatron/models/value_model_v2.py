# ======main:

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.global_vars import get_args
from megatron.global_vars import get_tokenizer
from models.base_model import BaseModel
from models.parallel_transformers import \
    ValueHead

from utils.utils import \
    get_advantages_and_returns
from .constants_ppo import \
    select_actions_from_right_padded


class ValueModel(BaseModel):

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 stats=None,
                 buffer=None):

        super(ValueModel, self).__init__(num_tokentypes,
                                         parallel_output,
                                         pre_process,
                                         post_process,
                                         add_pooler=True,
                                         pooler_cls=ValueHead)

        self.args = get_args()
        self.tokenizer = get_tokenizer()
        self.stats = stats
        self.buffer = buffer

    def forward(self, all_token_ids, all_position_ids, all_token_attention_mask, training_inputs=None,
                inference_params=None):
        lm_output = self.language_model(
            all_token_ids,
            all_position_ids,
            all_token_attention_mask,
            inference_params=inference_params)

        # if post process lm_output = hiddens, values
        # else: lm_output = hiddens

        # note in middle pipeline, this all_token_logits is just a hidden
        if self.post_process:
            hiddens, values = lm_output
            # is last pipeline stage, if inference return the last logits. if training, return the loss
            return self.post_language_model_processing(
                hiddens, training_inputs, values,
            )
        else:
            hiddens = lm_output
            return hiddens

    def post_language_model_processing(self, hiddens, training_inputs, values):

        # is last pipeline stage, if inference return the last logits. if training, return the loss

        inference_only = training_inputs is None
        # print(f"hiddens size: {hiddens.size()}")

        if inference_only:
            # [s b] => [b s ]
            print(f"value inference_only values size : {values.size()}")
            return values.transpose(0, 1).contiguous()  # [b,responses_szie]
        else:
            values_pred = values.transpose(0, 1).contiguous()  # [b, all_token size]

            old_values = training_inputs['action_values']
            old_rewards = training_inputs['action_rewards']
            response_length = old_rewards.shape[1]
            all_token_loss_mask = training_inputs["all_token_loss_mask"]
            # For a proper positional encoding in case of left padding
            advantages, returns = get_advantages_and_returns(self.args,
                                                             old_values, old_rewards, response_length
                                                             )

            advantages_nonzero_for_log = advantages.view(-1)[advantages.view(-1).nonzero()]

            self.stats["value/advantages_mean"] = advantages_nonzero_for_log.mean()
            self.stats["value/advantages_min"] = advantages_nonzero_for_log.min()
            self.stats["value/advantages_max"] = advantages_nonzero_for_log.max()
            self.stats["value/advantages_std"] = advantages_nonzero_for_log.std()

            returns_nonzero_for_log = returns.view(-1)[returns.view(-1).nonzero()]

            self.stats["value/returns_mean"] = returns_nonzero_for_log.mean()
            self.stats["value/returns_min"] = returns_nonzero_for_log.min()
            self.stats["value/returns_max"] = returns_nonzero_for_log.max()

            values_pred = values_pred[:, :-1]  # remove the last token since we want the head(hidden(state))

            values_pred = select_actions_from_right_padded(ts=values_pred,
                                                           action_starts=training_inputs["action_starts"] - 1,
                                                           response_size=response_length,
                                                           pad_value=0.0, dim=-1).contiguous()
            mask = select_actions_from_right_padded(ts=all_token_loss_mask,
                                                    action_starts=training_inputs["action_starts"] - 1,
                                                    response_size=response_length,
                                                    pad_value=0, dim=-1).contiguous()

            # start = training_inputs["action_start"]
            # end = training_inputs["action_end"]
            #
            # values_pred, mask = (
            #     values_pred[:, start:end],
            #     all_token_loss_mask[:, start:end],
            # )

            values_clipped = torch.clamp(
                values_pred,
                old_values - self.args.cliprange_value,
                old_values + self.args.cliprange_value,
            )
            n = mask.sum()

            vf_loss1 = (values_pred - returns) ** 2
            vf_loss2 = (values_clipped - returns) ** 2

            # value_target_var = torch.var(returns)
            # value_explained_var = 1 - torch.var(returns - values_pred) / value_target_var
            if self.args.clipped_value_only:
                vf_loss = (0.5 * vf_loss2) * mask
                vf_clipfrac = torch.sum((values_pred != values_clipped).float() * mask) / n

            else:
                vf_loss = (0.5 * torch.max(vf_loss1, vf_loss2)) * mask
                # vf_loss = 0.5 * vf_loss1 * mask
                vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n
            self.stats["value_model/vf_clipfrac"] = vf_clipfrac

            # forexplained var
            gathered_returns = [
                torch.zeros(returns.size(0), self.args.max_position_embeddings).to(torch.cuda.current_device()) for _ in
                range(torch.distributed.get_world_size(group=mpu.get_data_parallel_group()))]
            padded_returns = torch.zeros(returns.size(0), self.args.max_position_embeddings).to(
                torch.cuda.current_device())
            padded_returns[:, :returns.size(1)] = returns

            dist.all_gather(gathered_returns, padded_returns, group=mpu.get_data_parallel_group())

            gathered_value_preds = [
                torch.zeros(returns.size(0), self.args.max_position_embeddings).to(torch.cuda.current_device()) for _ in
                range(torch.distributed.get_world_size(group=mpu.get_data_parallel_group()))]
            padded_value_pred = torch.zeros(returns.size(0), self.args.max_position_embeddings).to(
                torch.cuda.current_device())
            padded_value_pred[:, :values_pred.size(1)] = values_pred

            dist.all_gather(gathered_value_preds, padded_value_pred, group=mpu.get_data_parallel_group())

            # RL related stats: global
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == (
                    torch.distributed.get_world_size() - 1):
                    self.buffer["value/returns"].extend(gathered_returns)
                    self.buffer["value/value_preds"].extend(gathered_value_preds)

            return vf_loss.contiguous()  # [b,response_size]
