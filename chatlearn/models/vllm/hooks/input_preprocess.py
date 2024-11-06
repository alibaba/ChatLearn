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
"""Hooks of vllm-0.6.3 input preprocess to pass prompt text."""


import inspect

# pylint: disable=unused-import,unused-argument
from vllm.inputs import preprocess


source = inspect.getsource(preprocess.InputPreprocessor._extract_prompt_components)
if 'parsed = parse_singleton_prompt(prompt)' in source:
    from vllm.inputs.parse import parse_singleton_prompt

    def extract_prompt_components(
        self,
        prompt,
        request_id,
        lora_request=None):
        '''
        Extract the components of any single encoder or decoder input prompt.

        Arguments:

        * request_id
        * prompt: single encoder or decoder input prompt
        * lora_request: this is only valid for decoder prompts

        Returns:

        * prompt
        * prompt_token_ids
        * multi_modal_data
        * mm_processor_kwargs (request-level input processor/mapper overrides)
        '''
        parsed = parse_singleton_prompt(prompt)

        assert parsed["type"] == "tokens", \
            f"you must pass prompt_token_ids when add request to scheduler. while prompt {prompt}"

        prompt_text = parsed["content"]["prompt"]
        prompt_token_ids = parsed["content"]["prompt_token_ids"]
        multi_modal_data = parsed["content"].get("multi_modal_data")
        mm_processor_kwargs = parsed["content"].get("mm_processor_kwargs")

        return (prompt_text, prompt_token_ids, multi_modal_data,
                mm_processor_kwargs)

    preprocess.InputPreprocessor._extract_prompt_components = extract_prompt_components
