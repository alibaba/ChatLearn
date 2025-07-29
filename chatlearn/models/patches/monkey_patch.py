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
"""Apply patches for different model architectures"""
def apply_sp_monkey_patch(model_config):
    print(f"applying sequence parallel patches for {model_config.architectures}")
    if model_config.architectures[0] == "Qwen2ForCausalLM":
        from chatlearn.models.patches.transformers.qwen2_patch import register_sp_attention_forward \
            # pylint: disable=import-outside-toplevel
        register_sp_attention_forward()
    elif model_config.architectures[0] == "Qwen3ForCausalLM":
        from chatlearn.models.patches.transformers.qwen3_patch import register_sp_attention_forward \
            # pylint: disable=import-outside-toplevel
        register_sp_attention_forward()
    else:
        raise ValueError(f"Unsupported model architecture: {model_config.architectures}")

def apply_group_gemm(model):
    print(f"applying groupgemm patches for {model.config.architectures[0]}")
    if model.config.architectures[0] == "Qwen3MoeForCausalLM":
        from chatlearn.models.patches.transformers.qwen3_moe_patch import apply_group_gemm_patch \
            # pylint: disable=import-outside-toplevel
        apply_group_gemm_patch(model)
    else:
        raise ValueError(f"Unsupported model architecture: {model.config.architectures} for groupgemm patch")
