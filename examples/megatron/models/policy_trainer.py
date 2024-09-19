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
"""policy trainer"""

from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from megatron.training import get_args
try:
    from megatron.training import get_num_microbatches
except ImportError:
    from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_tokenizer
from megatron.training import print_rank_0
from megatron.core import mpu
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.utils import calc_params_l2_norm
from megatron.training.utils import get_ltor_masks_and_position_ids

from chatlearn.utils import to_device
from .policy_model import PolicyModel as LegacyPolicyModel
from .mcore_policy_model import MCorePolicyModel
from .utils import training_log, get_eos_id, get_padding_length, pad_to_length
from .base_trainer import BaseTrainer
from .constants import TrainerEngine
from .constants import select_actions_from_right_padded, get_ltor_masks_and_position_ids_rlhf, pad_to_max_len


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
        self.loss_mean = 0.0
        self.acc_mean = 0.0

    def model_provider(self, pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building GPT model ...')
        if self.args.use_legacy_models:
            model = LegacyPolicyModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                stats=self.stats
            )
            if self.module_args.lora.enable_lora:
                from chatlearn.models.megatron.lora import convert_layer_to_lora # pylint: disable=import-outside-toplevel
                model = convert_layer_to_lora(model)
        else:
            model = MCorePolicyModel(
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                stats=self.stats
            )
            if self.module_args.lora.enable_lora:
                assert False, "ChatLearn do not support LoRA with Megatron-Core models currently."

        return model

    def get_dpo_batch(self, data_b):
        # TODO: move to ChatLearn framework later. add pad to max length config
        chosen_ids = to_device("cuda", data_b["chosen"])
        rejected_ids = to_device("cuda", data_b["rejected"])
        chosen_mask = to_device("cuda", data_b["chosen_mask"])
        rejected_mask = to_device("cuda", data_b["rejected_mask"])
        prompt_id_lens = to_device("cuda", data_b["prompt_id_lens"])

        inputs = {
            "reference_chosen_logps": data_b["reference_chosen_logps"],
            "reference_rejected_logps": data_b["reference_rejected_logps"],
            "chosen": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected": rejected_ids,
            "rejected_mask": rejected_mask,
            "prompt_id_lens": prompt_id_lens
        }
        return inputs

    def get_rlhf_batch(self, data_b):
        # TODO: move to ChatLearn framework later. add pad to max length config
        all_token_ids_right_padded = pad_to_max_len(data_b["all_token_ids_right_padded"], self.args.seq_length,
                                                    pad_value=get_eos_id(get_tokenizer()))
        all_token_loss_mask = pad_to_max_len(data_b["loss_mask"], self.args.seq_length, pad_value=0)

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
        return inputs

    def get_online_dpo_batch(self, data_b):
        ref_logprobs = data_b["ref_logprobs"].float()
        no_padded_query_ids = data_b["no_padded_query_ids"].long()
        prompt_dicts = data_b["str_prompts"]
        old_logprobs = data_b["old_logprobs"].float()

        tokens_ = data_b["all_token_ids_right_padded"].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

        if tokens.size(0) <  self.args.train_to_compare_num_responses:
            num_queries = 1
            num_responses = torch.tensor([tokens.size(0)], dtype=torch.long)
        else:
            assert tokens.size(
                0) % self.args.train_to_compare_num_responses == 0, "need to %0 to process all response of a query"

            num_queries = int(tokens.size(0) / self.args.train_to_compare_num_responses)

            num_responses = torch.tensor([self.args.train_to_compare_num_responses] * num_queries, dtype=torch.long) # nq * 1

        for start in range(0, len(tokens), self.args.train_to_compare_num_responses):

            should_be_same = no_padded_query_ids[start:min(start + self.args.train_to_compare_num_responses, len(tokens))]
            assert (should_be_same == should_be_same[0]).all(), f"{should_be_same}, {should_be_same[0]}"
            assert (should_be_same == should_be_same[0]).all(), \
                f"{should_be_same}, {prompt_dicts[start:min(start + self.args.train_to_compare_num_responses, len(tokens))]}"

        rw_scores = torch.sum(data_b["action_rewards"], dim=-1).view(num_queries, -1) # nq * nr

        # TODO tianhang move to sayang's framework later. add pad to max length config
        tokens = pad_to_max_len(tokens, self.args.seq_length, pad_value=get_eos_id(get_tokenizer()))
        labels = pad_to_max_len(labels, self.args.seq_length, pad_value=get_eos_id(get_tokenizer()))
        all_token_loss_mask = pad_to_max_len(data_b["loss_mask"], self.args.seq_length, pad_value=0)
        action_starts = data_b['action_start_indices']
        # don't count loss on the prompt tokens (only count loss on the last prompt tokens) since last rpompt token gives the firs taction
        # thus loss_mask[0: action start - 1] = 0
        for i in range(all_token_loss_mask.size(0)):
            all_token_loss_mask[i, 0:action_starts[i]-1] = 0

        # sft is [1:] because we want loss on the last prompt token to generate the first action
        # because sft loss mask = 0 if in prompt else 1. thus [1:] means last prompt token = 1 before that is all 0
        # however, our loss mask is different and calculated above. Which is 1 for each all_token,
        # first stop token + 1: is 0. Then 0: action start ind - 1 is 0. thus last prompt = 1. action = 1.
        # but because last input tokne is gone, we don't need last loss mask also
        loss_mask = all_token_loss_mask.long()
        loss_mask[:, -1] = 0

        # Get the masks and position ids.
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            get_tokenizer().eod,
            self.args.reset_position_ids,
            self.args.reset_attention_mask,
            self.args.eod_mask_loss,
        )

        inputs = {
            "all_token_ids_right_padded": tokens, # padded to seqlen for seqparallel
            "all_token_attention_mask": attention_mask,
            "all_token_position_ids": position_ids,
            "loss_mask": loss_mask,
            "rw_scores":rw_scores,
            "labels":labels, # this is to get parallel version of -logprob using the NLL loss
            "num_responses":num_responses,
            "ref_logprobs":ref_logprobs, #b, before pad tokens.size(1) - 1
            "old_logprobs": old_logprobs  # b, before pad tokens.size(1) - 1
        }
        assert old_logprobs.size() == ref_logprobs.size(), f"{old_logprobs.size()} == {ref_logprobs.size()}"

        assert tokens.size() == loss_mask.size(), f"tokens size: {tokens.size()}, loss_mask size: {loss_mask.size()}"
        assert tokens.size(1) == self.args.seq_length, f"{tokens.size(1)} == {self.args.seq_length}"
        return inputs

    def get_grpo_batch(self, data_b):
        self._logger.info("get grpo batch")
        args = self.args
        ref_logprobs = data_b["ref_logprobs"].float()
        old_logprobs = data_b["old_logprobs"].float()
        # prompt_dicts = data_b["prompt_dicts"]

        tokens_ = data_b["all_token_ids_right_padded"].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

        if tokens.size(0) <  args.train_to_compare_num_responses:
            num_queries = 1
            num_responses = torch.tensor([tokens.size(0)], dtype=torch.long)
        else:
            assert tokens.size(
                0) % get_args().train_to_compare_num_responses == 0, "need to %0 to process all response of a query"
            num_queries = int(tokens.size(0) / args.train_to_compare_num_responses)

            num_responses = torch.tensor([args.train_to_compare_num_responses] * num_queries, dtype=torch.long) # nq * 1

        advantages = data_b["advantages"]
        tokens = pad_to_max_len(tokens, args.seq_length,
                                                    pad_value=get_tokenizer().eod)
        labels = pad_to_max_len(labels, args.seq_length,
                                                    pad_value=get_tokenizer().eod)
        all_token_loss_mask = pad_to_max_len(data_b["loss_mask"], args.seq_length, pad_value=0)
        action_starts = data_b['action_start_indices']

        # don't count loss on the prompt tokens (only count loss on the last prompt tokens) since last rpompt token gives the firs taction
        # thus loss_mask[0: action start - 1] = 0
        # seem not affect the pg loss
        for i in range(all_token_loss_mask.size(0)):
            all_token_loss_mask[i, 0:action_starts[i]-1] = 0

        # sft is [1:] because we want loss on the last prompt token to generate the first action
        # because sft loss mask = 0 if in prompt else 1. thus [1:] means last prompt token = 1 before that is all 0
        # however, our loss mask is different and calculated above. Which is 1 for each all_token,
        # first stop token + 1: is 0. Then 0: action start ind - 1 is 0. thus last prompt = 1. action = 1.
        # but because last input tokne is gone, we don't need last loss mask also
        loss_mask = all_token_loss_mask.long()
        loss_mask[:, -1] = 0 #?

        # Get the masks and position ids.
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            get_tokenizer().eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss,
        )


        inputs = {
            "all_token_ids_right_padded": tokens, # padded to seqlen for seqparallel
            "all_token_attention_mask": attention_mask,
            "all_token_position_ids": position_ids,
            "all_token_loss_mask": loss_mask,
            "advantages":advantages,
            "action_starts": data_b['action_start_indices'],
            "action_logprobs" : data_b["action_logprobs"].float(), #response size
            "num_responses":num_responses,
            "labels": labels,
            "ref_logprobs":ref_logprobs, #b, before pad tokens.size(1) - 1
            "old_logprobs": old_logprobs  # b, before pad tokens.size(1) - 1
        }
        assert old_logprobs.size() == ref_logprobs.size(), f"{old_logprobs.size()} == {ref_logprobs.size()}"
        assert tokens.size() == all_token_loss_mask.size(), f"tokens size: {tokens.size()}, loss_mask size: {all_token_loss_mask.size()}"
        assert tokens.size(1) == args.seq_length, f"{tokens.size(1)} == {args.seq_length}"
        return inputs


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
        assert isinstance(data_b, dict), data_b

        if args.trainer_engine == TrainerEngine.DPO:
            inputs = self.get_dpo_batch(data_b)
        elif args.trainer_engine == TrainerEngine.RLHF:
            inputs = self.get_rlhf_batch(data_b)
        elif args.trainer_engine == TrainerEngine.ONLINE_DPO:
            inputs = self.get_online_dpo_batch(data_b)
        elif args.trainer_engine == TrainerEngine.GRPO:
            inputs = self.get_grpo_batch(data_b)
        else:
            raise RuntimeError(f"Error trainer_engine {args.trainer_engine}, \
                expect one of {list(TrainerEngine)}.")

        for k, v in inputs.items():
            inputs[k] = to_device("cuda", v)
        return inputs

    def dpo_loss_fn(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        use_ipo = self.model_args.get("use_ipo", False)
        if use_ipo:
            losses = (logits - 1 / (2 * self.args.dpo_weight)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            label_smoothing = self.model_args.get("label_smoothing", 0.0)
            losses = (
                -F.logsigmoid(self.args.dpo_weight * logits) * (1 - label_smoothing)
                - F.logsigmoid(-self.args.dpo_weight * logits) * label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.args.dpo_weight * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.args.dpo_weight * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards

    def get_dpo_loss(self, inputs, losses):
        chosen_ids = inputs["chosen"].squeeze(1)
        reference_chosen_logps, reference_rejected_logps = inputs["reference_chosen_logps"], inputs["reference_rejected_logps"]
        chosen_logps, rejected_logps = losses[:chosen_ids.shape[0]], losses[chosen_ids.shape[0]:]
        preference_loss, chosen_reward, reject_reward = self.dpo_loss_fn(
            chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps)

        loss = preference_loss
        accuracy = (chosen_reward > reject_reward).float().mean().item()
        averaged_loss = average_losses_across_data_parallel_group([loss])
        self.loss_mean = 0.9 * self.loss_mean + 0.1 * loss.item()
        self.acc_mean = 0.9 * self.acc_mean + 0.1 * accuracy
        self.stats["accuracy"] = self.acc_mean
        self.stats["dpo_loss"] = self.loss_mean

        return loss, {'policy lm avg loss': averaged_loss[0], 'policy lm loss': loss}

    def get_online_dpo_loss(self, inputs, losses):
        loss_mask, rw_scores, num_responses, ref_logprobs, old_logprobs = inputs["loss_mask"], inputs["rw_scores"], \
                                                            inputs["num_responses"], inputs["ref_logprobs"], inputs["old_logprobs"]
        assert old_logprobs.size() == ref_logprobs.size(), f"{old_logprobs.size()} == {ref_logprobs.size()}"

        args = get_args()

        logprobs = -losses.float() # this loss is nll which is -logprob

        reference_logprobs = ref_logprobs.float()
        reference_logprobs = pad_to_max_len(reference_logprobs, args.seq_length, pad_value=0)

        old_all_logprobs = old_logprobs.float()
        old_all_logprobs = pad_to_max_len(old_all_logprobs, args.seq_length, pad_value=0)

        assert loss_mask.size(1) == args.seq_length, f"{loss_mask.size()}"

        logprobs = logprobs * loss_mask
        old_all_logprobs = old_all_logprobs * loss_mask
        reference_logprobs = reference_logprobs * loss_mask

        clamp_dpo_logprobs = self.model_args.get("clamp_dpo_logprobs", True)
        if clamp_dpo_logprobs:
            clamp_dpo_logprobs_min = self.model_args.get("clamp_dpo_logprobs_min", -1e10)
            clamp_dpo_logprobs_max = self.model_args.get("clamp_dpo_logprobs_max", 0)
            old_all_logprobs = torch.clamp(old_all_logprobs, min=clamp_dpo_logprobs_min, max=clamp_dpo_logprobs_max)
            logprobs = torch.clamp(logprobs, min=clamp_dpo_logprobs_min, max=clamp_dpo_logprobs_max)
            reference_logprobs = torch.clamp(reference_logprobs, min=clamp_dpo_logprobs_min, max=clamp_dpo_logprobs_max)


        length = loss_mask.sum(-1)

        scores = logprobs.sum(-1) / length
        reference_scores = reference_logprobs.sum(-1) / length

        idx = 0
        loss = 0
        avg_dpo_loss = 0

        for i, n in enumerate(num_responses):
            diff = (scores[idx:idx + n] - reference_scores[idx:idx + n]).unsqueeze(0) - (
                    scores[idx:idx + n] - reference_scores[idx:idx + n]).unsqueeze(-1)  # b * b
            rw_score = rw_scores[i].squeeze(0)
            rw_diff = rw_score.unsqueeze(0) - rw_score.unsqueeze(-1)  # b * b

            a = torch.lt(rw_diff, 0).numel()
            b = rw_diff.numel()
            negative_indices = torch.nonzero(torch.lt(rw_diff, 0))
            c = len(negative_indices)
            batch_size = rw_diff.shape[0]
            num_zeros = batch_size * (batch_size - 1) / 2 - c
            self.stats['rw_diff_lt_0'] = a
            self.stats['rw_diff'] = b
            self.stats['negative_indices'] = c
            self.stats['negative_indices_ratio'] = float(c/a)
            self.stats['num_zeros'] = num_zeros
            self.stats['zero_ratio'] = float(num_zeros/a)

            if len(negative_indices) == 0:
                continue
            diff_transformed = diff[negative_indices[:, 0], negative_indices[:, 1]] * -1

            use_ipo = self.model_args.get("use_ipo", False)
            if use_ipo:
                dpo_loss = ((diff_transformed - 1 / (2 * args.dpo_weight)) ** 2).mean()
            else:
                dpo_loss = -F.logsigmoid(args.dpo_weight * diff_transformed).mean()
            if not torch.isnan(dpo_loss):
                avg_dpo_loss += dpo_loss
                loss += dpo_loss
                idx += n
        if loss == 0.0:
            loss = torch.tensor(0.0, device=logprobs.device, requires_grad=True)
        if avg_dpo_loss == 0.0:
            avg_dpo_loss = torch.tensor(0.0, device=logprobs.device)
        loss = loss / (len(num_responses) + 1e-5)
        avg_dpo_loss = avg_dpo_loss / (len(num_responses) + 1e-5)

        averaged_loss = average_losses_across_data_parallel_group([loss, avg_dpo_loss])
        return loss, {'lm_loss': averaged_loss[0], 'dpo_loss': averaged_loss[1]}

    def get_grpo_loss(self, inputs, losses):
        ppo_losses, kl_losses = losses #[b, response_size]

        old_logprobs = inputs['action_logprobs'] #[b, responses size]
        response_length = old_logprobs.shape[1]
        # Note the tkoken logits to get loss is only the actions. query doesn't have loss.
        action_loss_mask = select_actions_from_right_padded(ts=inputs["all_token_loss_mask"],
                                                            action_starts=inputs["action_starts"] - 1,
                                                            # because align iwth logits index
                                                            response_size=response_length,
                                                            pad_value=0,
                                                            dim=-1).contiguous()

        # action_loss_mask = all_token_loss_mask[:, start:end].contiguous() #[ b, response_size]
        action_loss_mask = action_loss_mask.view(-1).float()
        ppo_loss = torch.sum(ppo_losses.view(-1) * action_loss_mask) / action_loss_mask.sum()
        kl_loss = torch.sum(kl_losses.view(-1) * action_loss_mask) / action_loss_mask.sum()
        loss = ppo_loss + kl_loss * get_args().dpo_weight

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss, kl_loss])
        self.stats["policy/pg_loss"] = averaged_loss[0]
        self.stats["policy/kl_loss"] = averaged_loss[1]
        return loss, {'policy pg loss': averaged_loss[0], 'policy kl loss': averaged_loss[1]}


    def get_rlhf_loss(self, inputs, losses):
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

    def aggregate_loss_func(self, inputs, losses):  # [b, s]
        if self.args.trainer_engine == TrainerEngine.DPO:
            return self.get_dpo_loss(inputs, losses)
        elif self.args.trainer_engine == TrainerEngine.ONLINE_DPO:
            return self.get_online_dpo_loss(inputs, losses)
        elif self.args.trainer_engine == TrainerEngine.RLHF:
            return self.get_rlhf_loss(inputs, losses)
        elif self.args.trainer_engine == TrainerEngine.GRPO:
            return self.get_grpo_loss(inputs, losses)
        else:
            raise RuntimeError(f"unknown trainer engine {self.args.trainer_engine}, expect one of {list(TrainerEngine)}")

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids',
                which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        pad_value = get_eos_id(get_tokenizer())
        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])

        tp_size = self.tensor_model_parallel_size()
        sp_enabled = self.megatron_args.sequence_parallel

        max_length = get_padding_length(sp_enabled, tp_size, max_length)

        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, pad_value),
                pad_to_length(reject_ids, max_length, pad_value),
            ),
            dim=0,
        )

        max_length = max(c_mask.shape[1], r_mask.shape[1])
        max_length = get_padding_length(sp_enabled, tp_size, max_length)

        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks

    def _forward_step(self, batch_data, model):
        """Forward step."""
        inputs = self.get_batch(batch_data)
        if self.args.trainer_engine == TrainerEngine.DPO:
            chosen_ids = inputs["chosen"].squeeze(1)
            rejected_ids = inputs["rejected"].squeeze(1)
            chosen_mask = inputs["chosen_mask"].squeeze(1)
            rejected_mask = inputs["rejected_mask"].squeeze(1)
            prompt_id_lens = inputs["prompt_id_lens"]
            prompt_id_lens = torch.cat([prompt_id_lens, prompt_id_lens], dim=0)

            inputs_, attn_masks = self.concatenated_inputs(chosen_ids, chosen_mask, rejected_ids, rejected_mask)

            dpo_labels = inputs_.clone()
            tokens_ = inputs_[:, :]
            attention_mask, position_ids = get_ltor_masks_and_position_ids_rlhf(tokens_)
            losses = model.forward(all_token_ids=tokens_,
                                  all_position_ids=position_ids,
                                  all_token_attention_mask=attention_mask,
                                  training_inputs=inputs,
                                  inference_config={"DPO_labels":dpo_labels, "prompt_id_lens": prompt_id_lens, "orig_mask": attn_masks})
        elif self.args.trainer_engine in [TrainerEngine.ONLINE_DPO, TrainerEngine.RLHF, TrainerEngine.GRPO]:
            losses = model.forward(all_token_ids=inputs["all_token_ids_right_padded"],
                                all_position_ids=inputs["all_token_position_ids"],
                                all_token_attention_mask=inputs["all_token_attention_mask"],
                                training_inputs=inputs)
        else:
            raise RuntimeError(f"Error trainer_engine {self.args.trainer_engine}, expect one of {list(TrainerEngine)}.")


        return losses, partial(self.aggregate_loss_func,
                               inputs)  # will call loss_func(loss_mask, output_tensor) to get loss


    def post_update_stuffs(self, loss_dict, skipped_iter,
                           grad_norm, num_zeros_in_grad, iteration):
        # only last rank give kl coef.
        if torch.distributed.get_rank() == (
            torch.distributed.get_world_size() - 1):
            if self.args.trainer_engine == TrainerEngine.RLHF:
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
