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
"""Hooks of vllm-0.8.5 loader to load ckpt of megatron format."""


import torch

# pylint: disable=unused-import,wildcard-import,unused-argument,unexpected-keyword-arg,no-value-for-parameter
from vllm.model_executor.model_loader import loader
from vllm.model_executor.model_loader.loader import device_loading_context, _initialize_model
from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models import llama
from vllm.model_executor.models import qwen2, qwen2_moe
from vllm.config import VllmConfig

from chatlearn.utils.vllm_import_helper import LlamaForCausalLM
from chatlearn.utils.vllm_import_helper import QWenLMHeadModel
from chatlearn.utils.vllm_import_helper import Qwen2ForCausalLM
from chatlearn.utils.vllm_import_helper import Qwen2MoeForCausalLM
from chatlearn.utils.vllm_import_helper import get_model_architecture
from chatlearn.utils.utils import get_use_legacy_models

from chatlearn.utils.vllm_utils import (
    convert_llama_state_dict_from_megatron_to_vllm,
    convert_llama_state_dict_from_mcore_to_vllm,
    convert_qwen_state_dict_from_megatron_to_vllm,
    load_checkpoint
)

def load_weights(self, model_args):
    torch.distributed.barrier()
    self.model_args = model_args
    load_checkpoint(self, None, None, model_args=model_args)
    torch.distributed.barrier()

def load_state_dict(self, state_dict, strict=True, assign=False):
    qwen_version = None
    if isinstance(self, LlamaForCausalLM):
        use_legacy_models = get_use_legacy_models(self.model_args)
        if use_legacy_models:
            convert_state_dict_internal = convert_llama_state_dict_from_megatron_to_vllm
        else:
            convert_state_dict_internal = convert_llama_state_dict_from_mcore_to_vllm
    elif isinstance(self, QWenLMHeadModel):
        qwen_version = 1.0
        convert_state_dict_internal = convert_qwen_state_dict_from_megatron_to_vllm
    elif isinstance(self, Qwen2ForCausalLM) or (Qwen2MoeForCausalLM is not None and isinstance(self, Qwen2MoeForCausalLM)):
        qwen_version = 2.0
        convert_state_dict_internal = convert_qwen_state_dict_from_megatron_to_vllm
    else:
        raise RuntimeError(f"Unsupported model for vllm backend. \
            support [LlamaForCausalLM, QWenLMHeadModel, Qwen2ForCausalLM, Qwen2MoeForCausalLM] only, while {self}")

    state_dict = convert_state_dict_internal(self.model_args, self.config, qwen_version=qwen_version)
    super(type(self), self).load_state_dict(state_dict, strict=strict)


def init(self, load_config):
    # remove 'Model loader extra config' assert.
    self.load_config = load_config

loader.DummyModelLoader.__init__ = init

# add ckpt loading of megatron format
def load_model(self, vllm_config: VllmConfig):# -> nn.Module:
    device_config = vllm_config.device_config
    model_config = vllm_config.model_config
    target_device = torch.device(device_config.device)
    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            model = _initialize_model(vllm_config=vllm_config)
        # NOTE(woosuk): For accurate performance evaluation, we assign
        # random values to the weights.
        if self.load_config.model_loader_extra_config.get("need_load_ckpt", True) and \
                self.load_config.model_loader_extra_config.get("load", None) is not None:
            qwen2.Qwen2ForCausalLM.load_state_dict = load_state_dict
            qwen2.Qwen2ForCausalLM.load_weights = load_weights
            qwen2_moe.Qwen2MoeForCausalLM.load_state_dict = load_state_dict
            qwen2_moe.Qwen2MoeForCausalLM.load_weights = load_weights
            llama.LlamaForCausalLM.load_state_dict = load_state_dict
            llama.LlamaForCausalLM.load_weights = load_weights
            model.load_weights(self.load_config.model_loader_extra_config)
        else:
            initialize_dummy_weights(model)

        loader._process_weights_after_loading(model, model_config, target_device)
    return model.eval()

loader.DummyModelLoader.load_model = load_model
