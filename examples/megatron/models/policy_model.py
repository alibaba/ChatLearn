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
"""policy model"""

from megatron import get_args
from megatron.global_vars import get_tokenizer
from megatron.model.gpt_model import GPTModel
from megatron.model.language_model import parallel_lm_logits

from chatlearn.models.megatron.ops.policy_gradient import tensor_decomp_pg_loss
from .constants_ppo import select_actions_from_right_padded
from .utils import get_advantages_and_returns
from .utils import has_config_in_args
from .utils import get_eos_id


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
            from megatron.arguments import core_transformer_config_from_args # pylint: disable=import-outside-toplevel
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
                inference_params=None):
        hiddens = self.forward_lm(all_token_ids, all_position_ids,
                                  all_token_attention_mask, inference_params=inference_params)  # [b, s, v]
        # note in middle pipeline, this all_token_logits is just a hidden
        if self.post_process:
            # is last pipeline stage, if inference return the last logits. if training, return the loss
            return self.post_language_model_processing(
                hiddens, training_inputs,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output)
        else:
            return hiddens

    def post_language_model_processing(self, hiddens, training_inputs, logit_weights,
                                       parallel_output):

        # is last pipeline stage, if inference return the last logits. if training, return the loss

        inference_only = training_inputs is None

        # Output. Format [s b v]
        all_token_logits = parallel_lm_logits(
            hiddens,
            logit_weights,
            parallel_output)
        all_token_logits = all_token_logits.transpose(0, 1).contiguous()

        if inference_only:
            # [s b h] => [b s h]
            # TODO do we need to transpose????
            return all_token_logits
        else:
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
