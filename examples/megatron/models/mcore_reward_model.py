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
"""reward model"""

import torch
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import get_linear_layer
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)


class LinearPooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, config, score_dimensions):
        super().__init__(config=config)
        args = get_args()
        hidden_size = config.hidden_size
        init_method = config.init_method
        self.dense1 = get_linear_layer(hidden_size, hidden_size, init_method, args.perform_initialization)
        self.dense2 = get_linear_layer(hidden_size, score_dimensions, init_method, args.perform_initialization)
        self.sequence_parallel = args.sequence_parallel

    def forward(self, hidden_states, sequence_indices=None):
        # hidden_states: [s, b, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,  # [s, b, h]
                tensor_parallel_output_grad=False)

        if sequence_indices is not None:
            selected_hidden = torch.index_select(hidden_states, 0, sequence_indices)
            selected_hidden = selected_hidden.diagonal(dim1=0, dim2=1).T
            pooled = self.dense2(torch.nn.functional.relu(self.dense1(selected_hidden)))
        else:
            selected_hidden = hidden_states  # [s, b, h]
            pooled = self.dense2(torch.nn.functional.relu(self.dense1(selected_hidden))).squeeze(2)  # [s, b, scoredim]

        return pooled


class MCoreRewardModel(MCoreGPTModel):
    """RewardModel for MCore"""

    def __init__(self,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 pooler_head=LinearPooler,
                 score_dimension=1):
        self.args = get_args()
        use_te = self.args.transformer_impl == "transformer_engine"
        self.config = core_transformer_config_from_args(self.args)

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
            config=self.config,
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

        # Output
        if post_process:
            self.pooler_head = pooler_head(self.config,
                                           score_dimensions=score_dimension)
            self._pooler_head_key = 'pooler_head'
        else:
            self._pooler_head_key = None


    def _language_model_forward(self, input_ids=None, position_ids=None, attention_mask=None,
                                inference_params=None):
        if self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # TODO: CHECK!
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor?
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

    def forward(self, input_ids=None, position_ids=None, attention_mask=None,
                labels=None, inference_params=None,
                pooling_sequence_index=None,
                inference_config=None):
        lm_output = self._language_model_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inference_params=inference_params
        )

        if self.post_process:
            if inference_config is not None and "batch_encode" in inference_config:
                print('GPTrewrad model batch encoding, give the transformers encodings')
                if get_args().sequence_parallel:
                    lm_output = tensor_parallel.gather_from_sequence_parallel_region(
                        lm_output,  # [s, b, h]
                        tensor_parallel_output_grad=False)
                return lm_output
            assert labels is None, "assume labels is None in reward model"
            return self.pooler_head(lm_output, pooling_sequence_index)
        # [b x score_dim]
        return lm_output

    def load_state_dict(self, state_dict, strict=True):# pylint: disable=unused-argument
        """Customized load."""
        # Directly utilize super().load_state_dict(state_dict, strict=True) causes exceptions. This is
        # because if the base torch.nn.Module method load_state_dict is invoked, a strict key
        # matching mechanism will be enforced across all parameters. It will become
        # problematic when training a reward model derived from a SFT checkpoint, as
        # these checkpoints typically lack the state_dict for pooler_head in the reward model.
        incompatible_keys = super().load_state_dict(state_dict, strict=False)
        if len(incompatible_keys.missing_keys) == 0 and len(incompatible_keys.unexpected_keys) == 0:
            print_rank_0("load reward model pooler_head success")
            return
        elif self.post_process:
            if all(missing_key.startswith(self._pooler_head_key) for missing_key in incompatible_keys.missing_keys):
                print_rank_0("cannot load reward model pooler_head, init from random")
            if all(unexpected_key.startswith("output_layer") for unexpected_key in incompatible_keys.unexpected_keys):
                print_rank_0("neglect output_layer weight for reward model")
            return
        else:
            error_msgs: List[str] = []
            if len(incompatible_keys.unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join(f'"{k}"' for k in incompatible_keys.unexpected_keys)))
            if len(incompatible_keys.missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join(f'"{k}"' for k in incompatible_keys.missing_keys)))
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                            self.__class__.__name__, "\n\t".join(error_msgs)))


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = MCoreRewardModel(
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        score_dimension=1,
    )
    return model
