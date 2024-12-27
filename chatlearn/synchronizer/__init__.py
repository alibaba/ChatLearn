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
"""synchronizer"""

from transformers import AutoConfig
from chatlearn.models.megatron_module import MegatronModule
from chatlearn.models.vllm_module import VLLMModule
from chatlearn.models.vllm_module_v2 import VLLMModuleV2
from chatlearn.runtime.dist_actor import DistModel
from .base import BaseSync
from .megatron_megatron import MegatronMegatronSync
from .megatron_vllm import MegatronVllmQWenSync, MegatronVllmQWen2Sync, MegatronVllmLlamaSync

def get_synchronizer(src_model, dst_model):
    assert isinstance(src_model, DistModel)
    assert isinstance(dst_model, DistModel)
    src_model = src_model.replicas[0].model
    dst_model = dst_model.replicas[0].model
    if isinstance(src_model, MegatronModule) and isinstance(dst_model, MegatronModule):
        return MegatronMegatronSync(src_model, dst_model)
    elif isinstance(src_model, MegatronModule) and isinstance(dst_model, (VLLMModule, VLLMModuleV2)):
        config_dir = dst_model.module_args.args_dict["tokenizer"]
        config =  AutoConfig.from_pretrained(config_dir)
        model_class_name = config.architectures[0]
        if model_class_name == "QWenLMHeadModel":
            return MegatronVllmQWenSync(src_model, dst_model)
        elif model_class_name in ["Qwen2ForCausalLM", "Qwen2MoeForCausalLM"]:
            return MegatronVllmQWen2Sync(src_model, dst_model)
        elif model_class_name == "LlamaForCausalLM":
            return MegatronVllmLlamaSync(src_model, dst_model)
        else:
            raise RuntimeError(
                f"Unsupported model {model_class_name}, Expect QWenLMHeadModel, Qwen2ForCausalLM, Qwen2MoeForCausalLM or LlamaForCausalLM.")
    else:
        return BaseSync(src_model, dst_model)
