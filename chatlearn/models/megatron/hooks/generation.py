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
"""Megatron generation with limit in min_prompt_length."""

import inspect
# pylint: disable=unused-import,wildcard-import
from chatlearn.utils.megatron_import_hook_helper import *
from chatlearn.utils.megatron_import_hook_helper import _build_attention_mask_and_position_ids
# pylint: enable=unused-import,wildcard-import
from chatlearn.utils.utils import detect_and_insert_code_to_func


def limit_min_prompt_length(source_code):
    pattern = 'min_prompt_length = lengths.min().item()'
    new_code = \
"""
import chatlearn
if chatlearn.get_args().active_module_args.batch_generation.min_prompt_length:
    min_prompt_length = min(min_prompt_length, chatlearn.get_args().active_module_args.batch_generation.min_prompt_length)
"""
    source_code = detect_and_insert_code_to_func(source_code, pattern, new_code, line_offset=1)
    return source_code

source = inspect.getsource(generation.generate_tokens_probs_and_return_on_first_stage)
if 'batch_generation.min_prompt_length' not in source:
    exec(limit_min_prompt_length(source)) # pylint: disable=exec-used
    generation.generate_tokens_probs_and_return_on_first_stage = generate_tokens_probs_and_return_on_first_stage
