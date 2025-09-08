# pylint: disable=unused-argument,missing-class-docstring,abstract-method,invalid-overridden-method
# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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
"""sglang rollout"""
from typing import Any, Dict, List

import torch
import numpy as np
from transformers import AutoTokenizer

from chatlearn.configs import BaseConfig
from chatlearn.data.prompt_dataset import PromptPipeline
from chatlearn.models.sglang_module import AsyncSGLangModule, SGLangModule
from chatlearn.runtime.decorator import compute_decorator, timeit


def build_dataset_func(
    cfg: BaseConfig, tokenizer: AutoTokenizer, prompts: List[Dict], is_eval=False
):
    max_prompt_tokens_length = cfg.max_prompt_tokens_length

    prompts_dataset = PromptPipeline(
        prompts,
        max_prompt_tokens_length,
        tokenizer,
        enable_thinking=cfg.enable_thinking,
    )
    return prompts_dataset


def sglang_postprocess_func(
    tokenizer: AutoTokenizer,
    batched_outputs: List[Dict[str, Any]],
    input_data_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    data_output = []
    for output, input_data in zip(batched_outputs, input_data_list):
        prompt_token_ids = input_data["input_ids"]
        output_tokens = output["output_ids"]
        response_token_length = output["meta_info"]["completion_tokens"]
        str_outputs = tokenizer.decode(output_tokens, skip_special_tokens=True)
        all_tokens = torch.tensor(prompt_token_ids + output_tokens)
        input_data.update(
            {
                "all_tokens": all_tokens,
                "response_token_length": response_token_length,
                "str_outputs": str_outputs,
            }
        )
        if "rollout_round" in input_data:
            input_data["rollout_round"] += 1
        data_output.append(input_data)

    print("str_outputs", data_output[0]["str_outputs"])
    print("data_sources", data_output[0]["data_source"])
    print("ground_truth", data_output[0]["ground_truth"])
    return data_output


def metric_collect(rets, max_response_tokens_length):
    # collect metric
    response_token_length = [ret["response_token_length"] for ret in rets]
    prompt_token_length = [ret["prompt_token_length"] for ret in rets]
    clip_ratio = sum(
        ret["response_token_length"] >= ret.get("max_generate_token_length", max_response_tokens_length) \
            for ret in rets
    ) / len(rets)
    response_token_length.sort()
    inference_stats = {
        "response_token_length": sum(response_token_length)
        / len(response_token_length),
        "prompt_token_length": sum(prompt_token_length) / len(prompt_token_length),
        "response_clip_ratio": clip_ratio,
        "response_max": max(response_token_length),
        "response_25_percentile": np.percentile(response_token_length, 25),
        "response_50_percentile": np.percentile(response_token_length, 50),
        "response_75_percentile": np.percentile(response_token_length, 75),
    }
    return inference_stats


class SGLangPolicyInference(SGLangModule):
    """sglang rollout"""

    def build_dataset(self, prompts: List[Dict], is_eval=False):
        # TODO: move dataset to seperate node
        prompts_dataset= build_dataset_func(self.module_args, self.tokenizer, prompts, is_eval)
        self._logger.info(f"Max prompt token in data: {prompts_dataset.max_prompt}, valid data ratio: {prompts_dataset.valid_ratio}")
        return prompts_dataset

    @compute_decorator(trainable=False, rollout=True)
    @timeit()
    def eval_forward(self, data, iteration=0, **kwargs):
        return self._forward_step(data, iteration, True)

    def _forward_step(
        self, data, iteration, is_eval
    ):
        outputs = self.generate(data, is_eval)

        if outputs is not None:
            rets = sglang_postprocess_func(self.tokenizer, outputs, data)
            return rets

    @compute_decorator(trainable=False, rollout=True)
    @timeit()
    def forward_step(
        self, data: List[Dict[str, Any]], iteration=0, **kwargs
    ) -> List[Dict[str, Any]]:

        rets = self._forward_step(data, iteration, False)
        # collect metric
        self._metric_list.append(metric_collect(rets, self.module_args.max_response_tokens_length))
        return rets


class AsyncSGLangPolicyInference(AsyncSGLangModule):

    def build_dataset(self, prompts: List[Dict], is_eval=False):
        prompts_dataset= build_dataset_func(self.module_args, self.tokenizer, prompts, is_eval)
        self._logger.info(f"Max prompt token in data: {prompts_dataset.max_prompt}, valid data ratio: {prompts_dataset.valid_ratio}")
        return prompts_dataset

    @compute_decorator(trainable=False, rollout=True)
    @timeit()
    async def eval_forward(self, data, iteration=0, **kwargs):
        return await self._forward_step(data, iteration, True)

    async def _forward_step(
        self, data, iteration, is_eval
    ):  # pylint: disable=unused-argument
        outputs = await self.generate(data, is_eval)

        if outputs is not None:
            rets = sglang_postprocess_func(self.tokenizer, outputs, data)
            return rets

    @compute_decorator(trainable=False, rollout=True)
    @timeit()
    async def forward_step(
        self, data: List[Dict[str, Any]], iteration=0, **kwargs
    ) -> List[Dict[str, Any]]:  # pylint: disable=unused-argument
        rets = await self._forward_step(data, iteration, False)
        # collect metric
        self._metric_list.append(metric_collect(rets, self.module_args.max_response_tokens_length))
        return rets
