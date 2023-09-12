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
"""reward model"""

import torch
from dataset.reward_dataset import preprocess
from megatron import get_args
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.core import tensor_parallel
from megatron.model import GPTModel
from megatron.model.module import MegatronModule
from megatron.model.utils import get_linear_layer


def batch_padded_tokenize_data(list_strs, tokenizer, max_length):
    processed_dict = [preprocess(tokenizer.tokenize(line[0]), tokenizer.tokenize(line[1]), max_length, tokenizer) for
                      line in list_strs]
    input_ids, input_lengths = [], []
    for item in processed_dict:
        input_ids.append(torch.tensor(item['ids']))
        input_lengths.append(item['length'])
    max_l = min(max(input_lengths), max_length)
    input_ids = torch.stack(input_ids, dim=0)[:, :max_l]
    input_eos_tok = torch.tensor(input_lengths) - 1

    return input_ids, input_eos_tok


class LinearPooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method, score_dimensions):
        super().__init__()
        args = get_args()
        self.dense1 = get_linear_layer(hidden_size, hidden_size, init_method)
        self.dense2 = get_linear_layer(hidden_size, score_dimensions, init_method)
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


class RewardModel(GPTModel):
    """RewardModel"""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 pooler_head=LinearPooler,
                 score_dimension=1):
        super().__init__(num_tokentypes=num_tokentypes,
                         parallel_output=parallel_output,
                         pre_process=pre_process,
                         post_process=post_process)

        if self.post_process:
            self.pooler_head = pooler_head(self.language_model.hidden_size, self.language_model.init_method,
                                           score_dimensions=score_dimension)
            self._pooler_head_key = 'pooler_head'
        else:
            self._pooler_head_key = None
        self.tokenizer = get_tokenizer()

    def forward(self, input_ids=None, position_ids=None, attention_mask=None,
                ret_input_ids=None, ret_position_ids=None, ret_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None, # pylint: disable=unused-argument
                pooling_sequence_index=None,
                list_strs=None,  # pylint: disable=unused-argument
                ):
        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            ret_input_ids=ret_input_ids,
            ret_position_ids=ret_position_ids,
            ret_attn_mask=ret_attn_mask,
            inference_params=inference_params)
        if self.post_process:
            assert labels is None, "assume labels is None in reawrd model"
            return self.pooler_head(lm_output, pooling_sequence_index)
        # [b x score_dim]
        return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        state_dict_ = super().state_dict_for_save_checkpoint(prefix, keep_vars)

        if self.post_process:
            state_dict_[self._pooler_head_key] \
                = self.pooler_head.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        super().load_state_dict(state_dict, strict)
        if self._pooler_head_key in state_dict:
            # for rlhf training
            print_rank_0("load reward model pooler_head success")
            self.pooler_head.load_state_dict(state_dict[self._pooler_head_key], strict=strict)
        elif self.post_process:
            # for reward model training
            print_rank_0("cannot load reward model pooler_head, init from random")
