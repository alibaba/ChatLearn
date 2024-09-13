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
"""policy model"""

import torch

from megatron.training import get_args
from megatron.core import tensor_parallel
from megatron.training.global_vars import get_tokenizer
from megatron.legacy.model.gpt_model import GPTModel
from megatron.legacy.model.language_model import parallel_lm_logits

from chatlearn.models.megatron.ops.policy_gradient import tensor_decomp_pg_loss
from .utils import get_advantages_and_returns, has_config_in_args, get_eos_id
from .constants import TrainerEngine
from .constants import select_actions_from_right_padded



class PolicyModel(GPTModel):
    """PolicyModel"""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 stats=None):
        self.args = get_args()
        if has_config_in_args(GPTModel):
            # new API
            from megatron.training.arguments import core_transformer_config_from_args # pylint: disable=import-outside-toplevel
            config = core_transformer_config_from_args(self.args)
            super().__init__(config, num_tokentypes, parallel_output, pre_process, post_process)
        else:
            super().__init__(num_tokentypes, parallel_output, pre_process, post_process)
        self.tokenizer = get_tokenizer()
        self.stats = stats

    def forward_lm(self, input_ids, position_ids, attention_mask, inference_params=None):
        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            inference_params=inference_params)
        return lm_output

    def forward(self, all_token_ids, all_position_ids, all_token_attention_mask, training_inputs=None,
                inference_params=None, inference_config=None):
        hiddens = self.forward_lm(all_token_ids, all_position_ids,
                                  all_token_attention_mask, inference_params=inference_params)  # [b, s, v]
        # note in middle pipeline, this all_token_logits is just a hidden
        if self.post_process:
            # is last pipeline stage, if inference return the last logits. if training, return the loss
            use_parallel_output = inference_config["parallel_output"] if inference_config is not None and \
                "parallel_output" in inference_config else self.parallel_output

            if inference_config is not None and "DPO_labels" in inference_config:
                assert get_args().trainer_engine in [TrainerEngine.DPO.value, TrainerEngine.ONLINE_DPO.value]
                if training_inputs is None:
                    training_inputs = {}
                training_inputs["labels"] = inference_config["DPO_labels"]
                if get_args().trainer_engine == TrainerEngine.DPO:
                    assert "prompt_id_lens" in inference_config
                    assert "orig_mask" in inference_config
                    training_inputs["prompt_id_lens"] = inference_config["prompt_id_lens"]
                    all_token_attention_mask = inference_config["orig_mask"]
                    use_parallel_output = False
            return self.post_language_model_processing(
                hiddens, training_inputs,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                use_parallel_output,
                attention_mask=all_token_attention_mask)
        else:
            return hiddens

    def post_process_rlhf(self, training_inputs, all_token_logits):
        old_logprobs = training_inputs['action_logprobs']  # [b, responses size]
        old_values = training_inputs['action_values']  # [b, responses size]
        old_rewards = training_inputs['action_rewards']  # [b, responses size]
        response_length = old_rewards.shape[1]

        all_token_ids = training_inputs["all_token_ids_right_padded"]

        # For a proper positional encoding in case of left padding
        advantages, returns = get_advantages_and_returns(self.args,
                                                        old_values, old_rewards, response_length
                                                        )
        assert advantages.size(1) == returns.size(1) == response_length
        #    start = query_tensors.shape[1] - 1 for left padded
        #    end = action_start + response_length
        # Note the token logits to get loss is only the actions. query doesn't have loss.

        # all_token_ids = [pad, q1, q2, q3, a1, a2, a3, pad, pad]
        #                 [pad, q1, q2, q3, a1, a2, a3, a4, a5]
        # start = 4-1 = 3, end = 3 + 5 = 8
        # action_loss_mask = notpad(q3, a1, a2, a3, pad,]), notpad([q3, a1, a2, a3, a4], )
        # action_token_logits = logits(q3, a1, a2, a3, pad), logits(q3, a1, a2, a3, a4)
        # action_ids = [a1, a2, a3, pad, pad], [a1, a2, a3, a4, a5]

        action_loss_mask = select_actions_from_right_padded(ts=training_inputs["all_token_loss_mask"],
                                                            action_starts=training_inputs["action_starts"] - 1,
                                                            # because align iwth logits index
                                                            response_size=response_length,
                                                            pad_value=0, dim=-1).contiguous()

        # because we want the logits from the previous token
        # because it's -1 at top and then action -1 it hsould remain in bound
        action_token_logits = select_actions_from_right_padded(ts=all_token_logits[:, :-1, :],
                                                            action_starts=training_inputs["action_starts"] - 1,
                                                            response_size=response_length,
                                                            pad_value=1.0, dim=-2).contiguous()
        action_ids = select_actions_from_right_padded(ts=all_token_ids,
                                                    action_starts=training_inputs["action_starts"],
                                                    response_size=response_length,
                                                    pad_value=get_eos_id(self.tokenizer), dim=-1).contiguous()

        loss = tensor_decomp_pg_loss(self.args,
                                    action_token_logits=action_token_logits,  # [b,response size]
                                    action_ids=action_ids,  # [b, response size]
                                    action_loss_mask=action_loss_mask,  # [b, response size]
                                    old_logprobs=old_logprobs,  # [b, response size]
                                    advantages=advantages,  # [b, response size]
                                    stats=self.stats)  # [b, response_size] remove last logit because it's EOS

        self.approx_kl = self.stats["policy/approx_kl"]  # Update kl controller stats
        return loss.contiguous()  # [b,response_size]


    def post_process_dpo(self, logits, training_inputs, attention_mask, average_log_prob=False):
        assert "labels" in training_inputs and training_inputs['labels'] is not None
        labels =  training_inputs['labels']
        prompt_id_lens = training_inputs['prompt_id_lens']
        assert logits.shape[:-1] == labels.shape, \
            f"Mismatch tensor shape between logits.shape[:-1] ({logits.shape[:-1]}) and labels.shape ({labels.shape})"
        loss_masks = attention_mask.clone().bool()
        loss_masks = loss_masks.squeeze(1)
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        labels[loss_masks == False] = 0 # pylint: disable=singleton-comparison

        loss_masks = loss_masks[:, 1:]
        logits = logits[:, 1:, :]
        labels = labels[:, 1:]

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        else:
            return (per_token_logps * loss_masks).sum(-1)

    def post_process_online_dpo(self, sbv_all_token_logits, training_inputs):
        assert "labels" in training_inputs and training_inputs['labels'] is not None
        CE_loss = self.cross_entropy_loss(sbv_all_token_logits,
                                          training_inputs['labels'],
                                          self.args.fp16_lm_cross_entropy)
        return CE_loss

    def post_language_model_processing(self, hiddens, training_inputs, logit_weights,
                                       parallel_output, attention_mask=None):

        # is last pipeline stage, if inference return the last logits. if training, return the loss

        inference_only = training_inputs is None

        # Output. Format [s b v]
        all_token_logits = parallel_lm_logits(
            hiddens,
            logit_weights,
            parallel_output)

        sbv_all_token_logits = all_token_logits
        all_token_logits = all_token_logits.transpose(0, 1).contiguous()

        if inference_only:
            # [s b h] => [b s h]
            # TODO do we need to transpose????
            if self.args.trainer_engine == TrainerEngine.DPO:
                return self.post_process_dpo(all_token_logits, training_inputs, attention_mask)
            return all_token_logits
        else:
            if self.args.trainer_engine == TrainerEngine.DPO:
                return self.post_process_dpo(all_token_logits, training_inputs, attention_mask)
            elif self.args.trainer_engine == TrainerEngine.RLHF:
                return self.post_process_rlhf(training_inputs, all_token_logits)
            elif self.args.trainer_engine == TrainerEngine.ONLINE_DPO:
                return self.post_process_online_dpo(sbv_all_token_logits, training_inputs)
            elif self.args.trainer_engine == TrainerEngine.GRPO:
                return self.post_process_grpo(all_token_logits, sbv_all_token_logits, training_inputs)

    def post_process_grpo(self, all_token_logits, sbv_all_token_logits, training_inputs):
        all_token_ids = training_inputs["all_token_ids_right_padded"]
        adv_scores = torch.FloatTensor(training_inputs['advantages'])

        old_logprobs = training_inputs['action_logprobs'] #[b, responses size]
        response_length = old_logprobs.shape[1]
        action_loss_mask = select_actions_from_right_padded(ts=training_inputs["all_token_loss_mask"],
                                                            action_starts=training_inputs["action_starts"] - 1, # because align iwth logits index
                                                            response_size=response_length,
                                                            pad_value=0, dim=-1).contiguous()

        assert action_loss_mask.size(0) == len(adv_scores)

        self.stats["policy/adv_mean"] = adv_scores.mean()
        self.stats["policy/adv_std"] = adv_scores.std()

        adv = []
        for i, adv_score in enumerate(adv_scores):
            adv.append(adv_score * action_loss_mask[i].float())
        advantages = torch.stack(adv)
        assert advantages.size(0) == action_loss_mask.size(0)
        assert advantages.size(1) == action_loss_mask.size(1) == response_length

        # because we want the logits from the previous token
        # because it's -1 at top and then action -1 it hsould remain in bound [seem not true here]
        action_token_logits = select_actions_from_right_padded(ts=all_token_logits,
                                                            action_starts=training_inputs["action_starts"]-1,
                                                            response_size=response_length,
                                                                pad_value=1.0, dim=-2).contiguous()
        action_ids = select_actions_from_right_padded(ts=all_token_ids,
                                                            action_starts=training_inputs["action_starts"],
                                                            response_size=response_length,
                                                            pad_value=self.tokenizer.eod, dim=-1).contiguous()

        loss = tensor_decomp_pg_loss(self.args,
                                        action_token_logits=action_token_logits, # [b,response size]
                                        action_ids=action_ids,  # [b, response size]
                                        action_loss_mask=action_loss_mask,  # [b, response size]
                                        old_logprobs=old_logprobs,  # [b, response size]
                                        advantages=advantages,  # [b, response size]
                                        stats=self.stats)  # [b, response_size] remove last logit because it's EOS
        assert not torch.isnan(loss).any(), "pg loss is nan"
        #### KL Loss ####
        forward_logprob = self.cross_entropy_loss(sbv_all_token_logits,
                                training_inputs["labels"],
                                get_args().fp16_lm_cross_entropy) * -1
        ref_logprobs = training_inputs['ref_logprobs']

        action_forward_logprobs = select_actions_from_right_padded(ts=forward_logprob,
                                        action_starts=training_inputs["action_starts"]-1, # because align iwth logits index
                                        response_size=response_length,
                                        pad_value=0, dim=-1).contiguous()
        action_ref_logprobs = select_actions_from_right_padded(ts=ref_logprobs,
                                        action_starts=training_inputs["action_starts"]-1, # because align iwth logits index
                                        response_size=response_length,
                                        pad_value=0, dim=-1).contiguous()
        assert action_forward_logprobs.size(-1) == action_ref_logprobs.size(-1) == loss.size(-1)

        if get_args().numerical_stable:
            logprob_diff = torch.clamp(action_ref_logprobs - action_forward_logprobs, min=-1e5, max=1e5)
            log_ratio = (logprob_diff) * action_loss_mask
            # numerical approximate an exponential for numerical stability
            ratio = 1 + log_ratio + torch.square(log_ratio) / 2
        else:
            logprob_diff = action_ref_logprobs - action_forward_logprobs
            log_ratio = (logprob_diff) * action_loss_mask
            ratio = torch.exp(log_ratio)
        kl_loss = (ratio - log_ratio - 1).contiguous()
        assert not torch.isnan(loss).any(), "kl loss is nan"
        self.approx_kl = self.stats["policy/approx_kl"]  # Update kl controller stats
        return loss.contiguous(), kl_loss.contiguous()  # [b,response_size]

    def cross_entropy_loss(self, sbv_all_token_logits, labels, fp16_lm_cross_entropy):
        #all_token_logits is [s,b,vp]
        labels = labels.transpose(0, 1).contiguous() #[s,b]
        # if flash_cross_entropy is not None:
        #     loss = flash_cross_entropy(output.flatten(0, 1), labels.flatten()).view(*labels.size())

        if fp16_lm_cross_entropy:
            assert sbv_all_token_logits.dtype == sbv_all_token_logits.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(sbv_all_token_logits, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(sbv_all_token_logits.float(), labels)

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss
