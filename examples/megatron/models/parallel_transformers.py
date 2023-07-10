# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer based language model."""

import torch

from megatron import get_args
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.model.enums import LayerType, AttnMaskType
from megatron.model.language_model import Pooler, TransformerLanguageModel
from megatron.model.module import MegatronModule
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal, scaled_init_method_normal


def get_language_model(num_tokentypes, add_pooler, pooler_cls,
                       encoder_attn_mask_type, init_method=None,
                       scaled_init_method=None, add_encoder=True,
                       add_decoder=False,
                       decoder_attn_mask_type=AttnMaskType.causal,
                       pre_process=True, post_process=True):
    """Build language model and return along with the key to save."""
    args = get_args()

    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

    # Language model.
    language_model = RlhfTransformerLanguageModel(
        init_method,
        scaled_init_method,
        encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pooler_cls=pooler_cls,
        pre_process=pre_process,
        post_process=post_process
    )
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class ValueHead(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method):
        super(ValueHead, self).__init__()
        args = get_args()
        self.dense1 = get_linear_layer(hidden_size, 2 * hidden_size, init_method)  # split this after wards
        self.dense2 = get_linear_layer(2 * hidden_size, 1, init_method)
        self.sequence_parallel = args.sequence_parallel

    def forward(self, hidden_states):
        # hidden_states: [s, b, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(  # [s, b, h]
                hidden_states,
                tensor_parallel_output_grad=False)

        values = self.dense1(hidden_states)  # [s, b, 2h]
        values = torch.tanh(values)  # [s, b, 2h]
        values = self.dense2(values).squeeze(-1)  # [s, b, 1]

        return values  # [s, b]


def freeze_module(module, num_frozen_layers):
    if num_frozen_layers > 0:
        if module.offset is not None:
            for i in range(module.num_layers):
                layer_number = i + 1 + module.offset  # global layer number
                if layer_number < num_frozen_layers:
                    layer = module.layers[i]  # i is local layer number
                    layer.requires_grad_(False)
        else:
            print(f"it seems no transformer layer is present. only a no op one")


class RlhfTransformerLanguageModel(TransformerLanguageModel):

    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_encoder=True,
                 add_decoder=False,
                 decoder_attn_mask_type=AttnMaskType.causal,
                 add_pooler=False,
                 num_frozen_layers=0,
                 pooler_cls=Pooler,
                 pre_process=True,
                 post_process=True):
        super().__init__(init_method,
                         output_layer_init_method,
                         encoder_attn_mask_type,
                         num_tokentypes=num_tokentypes,
                         add_encoder=add_encoder,
                         add_decoder=add_decoder,
                         decoder_attn_mask_type=decoder_attn_mask_type,
                         add_pooler=False,  # add later
                         pre_process=pre_process,
                         post_process=post_process)
        self.args = get_args()
        self.num_frozen_layers = num_frozen_layers
        if self.encoder is not None and self.num_frozen_layers > 0:
            freeze_module(self.encoder, self.num_frozen_layers)
        if self.decoder is not None and self.num_frozen_layers > 0:
            freeze_module(self.decoder, self.num_frozen_layers)

        if self.post_process:
            # Pooler.
            if add_pooler:
                self.pooler = pooler_cls(self.hidden_size, self.init_method)
                if isinstance(self.pooler, Pooler):
                    self._pooler_key = 'pooler'
                else:
                    self._pooler_key = 'value_head'
        self.add_pooler = add_pooler

    def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask,
                dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None,
                ret_input_ids=None, ret_position_ids=None, ret_attn_mask=None,
                enc_dec_attn_mask=None, tokentype_ids=None,
                inference_params=None,
                pooling_sequence_index=0,
                enc_hidden_states=None, output_enc_hidden=False):

        # Retriever embedding.
        if self.retriever and self.pre_process:
            retriever_input = self.embedding(ret_input_ids, ret_position_ids,
                                             tokentype_ids=tokentype_ids)
        else:
            retriever_input = None

        # Encoder embedding.
        if self.pre_process:
            encoder_input = self.embedding(enc_input_ids, enc_position_ids,
                                           tokentype_ids=tokentype_ids)
        else:
            encoder_input = None

        # Rotary positional embeddings
        rotary_pos_emb = None
        if self.use_rotary_position_embeddings:
            if inference_params is not None:
                rotary_pos_emb = \
                    self.rotary_pos_emb(inference_params.max_sequence_len)
            else:
                rotary_pos_emb = self.rotary_pos_emb(self.seq_length)

        # Run encoder.
        if enc_hidden_states is None:
            if self.encoder is not None:
                if self.retriever:
                    encoder_output = self.encoder(
                        encoder_input,
                        enc_attn_mask,
                        retriever_output=retriever_input,
                        retriever_attn_mask=ret_attn_mask,
                        inference_params=inference_params)
                else:
                    encoder_output = self.encoder(
                        encoder_input,
                        enc_attn_mask,
                        inference_params=inference_params,
                        rotary_pos_emb=rotary_pos_emb)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.post_process:
            if self.add_pooler:
                if isinstance(self.pooler, Pooler):
                    pooled_output = self.pooler(encoder_output,
                                                pooling_sequence_index)
                else:
                    pooled_output = self.pooler(encoder_output)

        # output_enc_hidden refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(dec_input_ids,
                                           dec_position_ids)
        else:
            decoder_input = None

        # Run decoder.
        decoder_output = self.decoder(
            decoder_input,
            dec_attn_mask,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
            inference_params=inference_params)

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if '_embeddings' in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

        # Encoder.
        if self.add_encoder:
            if self._encoder_key in state_dict:
                state_dict_ = state_dict[self._encoder_key]
            # For backward compatibility.
            elif 'transformer' in state_dict:
                state_dict_ = state_dict['transformer']
            else:
                # For backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'transformer.' in key:
                        state_dict_[key.split('transformer.')[1]] = state_dict[key]

            # For backward compatibility.
            state_dict_self_attention = {}
            for key in state_dict_.keys():
                if '.attention.' in key:
                    state_dict_self_attention[key.replace(".attention.",
                                                          ".self_attention.")] = state_dict_[key]
                else:
                    state_dict_self_attention[key] = state_dict_[key]
            state_dict_ = state_dict_self_attention

            self.encoder.load_state_dict(state_dict_, strict=strict)

        # Pooler.
        if self.post_process:
            if self.add_pooler:
                if self.args.continue_train:
                    # tianhang: Only value function has pooler here. So DONT load pooler because it's the pooler of reward function init.
                    # UNLESS it's continue train where you want to load the last value function.
                    assert isinstance(self.pooler, ValueHead), f"type(self.pooler) {type(self.pooler)}"

                    self.pooler.load_state_dict(state_dict[self._pooler_key],
                                                strict=strict)
            if self.untie_embeddings_and_output_weights:
                assert 'output_layer' in state_dict, \
                    'could not find data for output_layer in the checkpoint'
                self.output_layer.load_state_dict(state_dict[self._output_layer_key],
                                                  strict=strict)
        # Decoder.
        if self.add_decoder:
            assert 'decoder' in state_dict, \
                'could not find data for pooler in the checkpoint'
            self.decoder.load_state_dict(state_dict[self._decoder_key],
                                         strict=strict)
