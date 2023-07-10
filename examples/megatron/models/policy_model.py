# ======main:


from megatron import get_args
from megatron.global_vars import get_tokenizer
from megatron.model.language_model import parallel_lm_logits
from models.base_model import BaseModel

from rlhf.models.ops.policy_gradient import tensor_decomp_pg_loss
from utils.utils import \
    get_advantages_and_returns
from .constants_ppo import select_actions_from_right_padded


class PolicyModel(BaseModel):

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 stats=None):

        super(PolicyModel, self).__init__(num_tokentypes,
                                          parallel_output,
                                          pre_process,
                                          post_process,
                                          add_pooler=False,
                                          pooler_cls=None)

        self.args = get_args()
        self.tokenizer = get_tokenizer()
        self.stats = stats

    def forward(self, all_token_ids, all_position_ids, all_token_attention_mask, training_inputs=None,
                inference_params=None):
        hiddens = self.forward_lm(all_token_ids, all_position_ids,
                                  all_token_attention_mask, inference_params=inference_params)  # [b, s, v]
        # note in middle pipeline, this all_token_logits is just a hidden
        if self.post_process:
            # is last pipeline stage, if inference return the last logits. if training, return the loss
            return self.post_language_model_processing(
                hiddens, training_inputs,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.word_embeddings_weight(),
                self.parallel_output)
        else:
            return hiddens

    def post_language_model_processing(self, hiddens, training_inputs, logit_weights,
                                       parallel_output):

        # is last pipeline stage, if inference return the last logits. if training, return the loss

        inference_only = training_inputs is None
        # print(f"hiddens size: {hiddens.size()}")

        # Output. Format [s b v]
        all_token_logits = parallel_lm_logits(
            hiddens,
            logit_weights,
            parallel_output)
        all_token_logits = all_token_logits.transpose(0, 1).contiguous()

        # print(f"all_token_logits size: {all_token_logits.size()}")

        if inference_only:
            # [s b h] => [b s h]
            # TODO do we need to transpose????
            return all_token_logits
        else:
            '''
            "all_token_position_ids": all_token_position_ids,
            "all_token_ids_right_padded" : data_b["all_token_ids_right_padded"],
            "all_token_attention_mask": all_token_attention_mask.bool(),
            "all_token_loss_mask": all_token_loss_mask.bool(),

            "action_starts": data_b['action_start_indices'],
            "action_logprobs" : data_b["action_logprobs"].float(), #response size
            "action_values" : data_b["action_values"].float(),
            "action_rewards" : data_b["action_rewards"].float(),
            '''
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
                                                          pad_value=self.tokenizer.eod_id, dim=-1).contiguous()

            # print(f"policy model beofre pg: training_inputs.size(): {training_inputs.size()}")
            # print(f"policy model beofre pg: action_ids.size(): {action_ids.size()}")
            # print(f"policy model beofre pg: advantages.size(): {advantages.size()}")
            # print(f"policy model beofre pg: old_logprobs.size(): {old_logprobs.size()}")
            # print(f"policy model beofre pg: action_loss_mask.size(): {action_loss_mask.size()}")
            # print(f"policy model beofre pg: action_token_logits.size(): {action_token_logits.size()}")

            loss = tensor_decomp_pg_loss(self.args,
                                         action_token_logits=action_token_logits,  # [b,response size]
                                         action_ids=action_ids,  # [b, response size]
                                         action_loss_mask=action_loss_mask,  # [b, response size]
                                         old_logprobs=old_logprobs,  # [b, response size]
                                         advantages=advantages,  # [b, response size]
                                         stats=self.stats)  # [b, response_size] remove last logit because it's EOS
            # remove first token_ids because it's starting EOS
            #
            # if loss.sum() > 10:
            #     print(f"loss out of roofff")
            #     print(f"---------old_values: {old_values}")
            #     print(f"---------old_rewards: {old_rewards}")
            #     print(f"---------advantages: {advantages}")
            #

            self.approx_kl = self.stats["policy/approx_kl"]  # Update kl controller stats
            return loss.contiguous()  # [b,response_size]
