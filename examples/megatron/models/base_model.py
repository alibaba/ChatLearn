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

"""GPT-2 model."""

from megatron import get_args
from megatron.model.enums import AttnMaskType
from megatron.model.gpt_model import GPTModel
from megatron.model.language_model import Pooler
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from models.parallel_transformers import \
    get_language_model


class BaseModel(GPTModel):
    """GPT-2 Language model."""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 add_pooler=False,
                 pooler_cls=Pooler):
        super(BaseModel, self).__init__(num_tokentypes, parallel_output, pre_process, post_process)
        args = get_args()
        self.share_word_embeddings = not args.untie_embeddings_and_output_weights

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=add_pooler,
            pooler_cls=pooler_cls,
            encoder_attn_mask_type=AttnMaskType.causal,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers),
            pre_process=self.pre_process,
            post_process=self.post_process)

        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings(init_method_normal)

    def forward_lm(self, input_ids, position_ids, attention_mask, inference_params=None):
        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            inference_params=inference_params)
        return lm_output
