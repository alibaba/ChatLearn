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
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.legacy.model.language_model import parallel_lm_logits
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

from chatlearn.models.megatron.ops.policy_gradient import tensor_decomp_pg_loss
from .utils import get_advantages_and_returns, get_eos_id
from .constants import TrainerEngine
from .constants import select_actions_from_right_padded


class MCorePolicyModel(MCoreGPTModel):
    """PolicyModel for MCore"""

    def __init__(self,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 stats=None):
        self.args = get_args()
        use_te = self.args.transformer_impl == "transformer_engine"
        config = core_transformer_config_from_args(self.args)

        if self.args.spec is not None:
            transformer_layer_spec = import_module(self.args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    self.args.num_experts,
                    self.args.moe_grouped_gemm,
                    self.args.qk_layernorm
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    self.args.num_experts,
                    self.args.moe_grouped_gemm,
                    self.args.qk_layernorm
                )

        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=self.args.padded_vocab_size,
            max_sequence_length=self.args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=self.args.fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=not self.args.untie_embeddings_and_output_weights,
            position_embedding_type=self.args.position_embedding_type,
            rotary_percent=self.args.rotary_percent,
            rotary_base=self.args.rotary_base
        )

        self.tokenizer = get_tokenizer()
        self.stats = stats

    def forward_lm(self, input_ids, position_ids, attention_mask, inference_params=None):
        if self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        lm_output = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )

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
                assert get_args().trainer_engine in [TrainerEngine.DPO, TrainerEngine.ONLINE_DPO]
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
                self.shared_embedding_or_output_weight() if self.share_embeddings_and_output_weights else self.output_layer.weight,
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
            elif self.args.trainer_engine == TrainerEngine.RLHF.value:
                return self.post_process_rlhf(training_inputs, all_token_logits)
            elif self.args.trainer_engine == TrainerEngine.ONLINE_DPO:
                return self.post_process_online_dpo(sbv_all_token_logits, training_inputs)

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
