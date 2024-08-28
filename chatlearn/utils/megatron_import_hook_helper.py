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
""""Version compatibility for hook"""

# pylint: disable=unused-import,wildcard-import

# megatron.text_generation.*
try:
    from megatron.text_generation import generation
    from megatron.text_generation.generation import *
    from megatron.text_generation.generation import _build_attention_mask_and_position_ids
    from megatron.text_generation.generation import generate_tokens_probs_and_return_on_first_stage
except ImportError:
    from megatron.inference.text_generation import generation
    from megatron.inference.text_generation.generation import *
    from megatron.inference.text_generation.generation import _build_attention_mask_and_position_ids
    from megatron.inference.text_generation.generation import generate_tokens_probs_and_return_on_first_stage

# pylint: enable=unused-import,wildcard-import
