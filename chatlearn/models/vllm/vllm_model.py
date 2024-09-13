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
"""vllm-based model"""

import torch
from torch import nn

from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion

from chatlearn.utils.vllm_import_helper import LlamaForCausalLM
from chatlearn.utils.vllm_import_helper import QWenLMHeadModel
from chatlearn.utils.vllm_import_helper import Qwen2ForCausalLM
from chatlearn.utils.vllm_import_helper import get_model_architecture

from chatlearn.utils.vllm_utils import (
    convert_llama_state_dict_from_megatron_to_vllm,
    convert_qwen_state_dict_from_megatron_to_vllm,
    load_checkpoint
)


class VLLMModel(nn.Module):
    """VLLM based Model"""

    def __init__(self, config, model_args, cache_config, quant_config, lora_config):
        super().__init__()
        self.config = config
        self.model_args = model_args
        self.model_class = get_model_architecture(config)
        if CURRENT_VLLM_VERSION == VLLMVersion.v_0_3_0.value:
            self.model = self.model_class(config.hf_config)
        elif CURRENT_VLLM_VERSION == VLLMVersion.v_0_5_1.value:
            self.model = self.model_class(config.hf_config, cache_config, quant_config, lora_config)

    def load_weights(self):
        torch.distributed.barrier()
        load_checkpoint(self, None, None)
        torch.distributed.barrier()

    def load_state_dict(self, state_dict, strict=True, assign=False): # pylint: disable=unused-argument
        qwen_version = None
        if isinstance(self.model, LlamaForCausalLM):
            convert_state_dict_internal = convert_llama_state_dict_from_megatron_to_vllm
        elif isinstance(self.model, QWenLMHeadModel):
            qwen_version = 1.0
            convert_state_dict_internal = convert_qwen_state_dict_from_megatron_to_vllm
        elif isinstance(self.model, Qwen2ForCausalLM):
            qwen_version = 2.0
            convert_state_dict_internal = convert_qwen_state_dict_from_megatron_to_vllm
        else:
            raise RuntimeError(f"Unsupported model for vllm backend. \
                support [LlamaForCausalLM, QWenLMHeadModel, Qwen2ForCausalLM] only, while {self.model_class}")

        state_dict = convert_state_dict_internal(self.model_args, self.config.hf_config, qwen_version=qwen_version)
        super().load_state_dict(state_dict, strict=strict)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
