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
from vllm.inputs.data import token_inputs

source = inspect.getsource(preprocess.InputPreprocessor._prompt_to_llm_inputs)
if 'parsed = parse_singleton_prompt(prompt)' in source:
    from vllm.inputs.parse import parse_singleton_prompt


    def _prompt_to_llm_inputs(
        self,
        prompt,
        request_id: str,
        lora_request=None,
    ):
        """
        Extract the singleton inputs from a prompt.

        Arguments:

        * request_id
        * prompt: single encoder or decoder input prompt
        * lora_request: this is only valid for decoder prompts

        Returns:

        * :class:`SingletonInputs` instance
        """
        parsed = parse_singleton_prompt(prompt)

        assert parsed["type"] == "tokens", \
            f"you must pass prompt_token_ids when add request to scheduler. while prompt {prompt}"

        if parsed["type"] == "tokens":
            tokens_content = parsed["content"]

            prompt_token_ids = tokens_content["prompt_token_ids"]
            token_type_ids = tokens_content.get("token_type_ids")
            multi_modal_data = tokens_content.get("multi_modal_data")
            mm_processor_kwargs = tokens_content.get("mm_processor_kwargs")

            if multi_modal_data is not None and self._can_process_multimodal():
                return self._process_multimodal(
                    prompt_token_ids,
                    multi_modal_data,
                    mm_processor_kwargs,
                    lora_request=lora_request,
                )

            return token_inputs(
                prompt=tokens_content["prompt"],
                prompt_token_ids=prompt_token_ids,
                token_type_ids=token_type_ids,
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
            )

    preprocess.InputPreprocessor._prompt_to_llm_inputs = _prompt_to_llm_inputs
