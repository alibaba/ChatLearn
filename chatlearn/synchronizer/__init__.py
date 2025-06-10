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
from typing import TYPE_CHECKING
from transformers import AutoConfig
from chatlearn.models.megatron_module import MegatronModule
from chatlearn.models.vllm_module import VLLMModule
from .megatron_vllm import(
    MegatronVllmMoonlightSync,
    MegatronVllmQWen2MCoreSync
)

if TYPE_CHECKING:
    from chatlearn.runtime.dist_actor import DistModel

def get_synchronizer(src_model: 'DistModel', dst_model: 'DistModel'):
    src_model = src_model.replicas[0].model
    dst_model = dst_model.replicas[0].model
    if not (isinstance(src_model, MegatronModule) and isinstance(dst_model, VLLMModule)):
        raise NotImplementedError(f"Do not support parameter synchronization between {type(src_model)} and {type(dst_model)}.")
    # NOTE: the parameter sync of megatron-vllm model pairs are also removed.
    name_to_synchronizer = {
        "Qwen2ForCausalLM": MegatronVllmQWen2MCoreSync,
        "Qwen2MoeForCausalLM": MegatronVllmQWen2MCoreSync,
        "DeepseekV3ForCausalLM": MegatronVllmMoonlightSync,
        "Qwen3ForCausalLM": MegatronVllmMoonlightSync,
        "Qwen3MoeForCausalLM": MegatronVllmMoonlightSync
    }
    config_dir = dst_model.module_args.args_dict["tokenizer"]
    config =  AutoConfig.from_pretrained(config_dir, trust_remote_code=True)
    model_class_name = config.architectures[0]
    if model_class_name not in name_to_synchronizer:
        raise RuntimeError(f"Unsupported model {model_class_name}, currently support {list(name_to_synchronizer.keys())}.")
    return name_to_synchronizer[model_class_name](src_model, dst_model)
