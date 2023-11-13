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
"""vllm policy model"""

import torch
from torch import nn

from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.qwen import QWenLMHeadModel
from vllm.model_executor.model_loader import _get_model_architecture

from chatlearn.utils.vllm_utils import (
    convert_lamma_state_dict_from_megatron_to_vllm,
    convert_qwen_state_dict_from_megatron_to_vllm,
    load_checkpoint
)


class VLLMPolicyModel(nn.Module):
    """VLLMPolicyModel"""

    def __init__(self, config, model_args):
        super().__init__()
        self.config = config
        self.model_args = model_args
        self.model_class = _get_model_architecture(config.hf_config)
        self.model = self.model_class(config.hf_config)

    def load_weights(self):
        torch.distributed.barrier()
        load_checkpoint(self, None, None)

    def load_state_dict(self, state_dict, strict=True):
        if isinstance(self.model, LlamaForCausalLM):
            convert_state_dict_internal = convert_lamma_state_dict_from_megatron_to_vllm
        elif isinstance(self.model, QWenLMHeadModel):
            convert_state_dict_internal = convert_qwen_state_dict_from_megatron_to_vllm
        else:
            raise RuntimeError(f"Unsupported model for vllm backend. \
                support LlamaForCausalLM only, while {self.model_class}")

        state_dict = convert_state_dict_internal(self.model_args, self.config.hf_config)
        super().load_state_dict(state_dict, strict=strict)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
