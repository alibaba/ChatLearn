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
from typing import Dict, List

import torch
import torch.nn.functional as F

from chatlearn.data.prompt_dataset import PromptPipeline
from chatlearn.models.sglang_module import SGLangModule



class SGLangPolicyInference(SGLangModule):
    """sglang rollout"""
    # pylint: disable=abstract-method
    def build_dataset(self, prompts: List[Dict], is_eval=False):
        # prompts seems like the total data set by engine.set_dataset(dataset)
        seq_length = self.module_args.get("seq_length")

        prompts_dataset = PromptPipeline(
            prompts,
            seq_length,
            self.tokenizer.tokenizer,
            enable_thinking=self.module_args.get("enable_thinking", False),
        )

        return prompts_dataset

    def eval_forward(self, data, iteration=0):
        return self._forward_step(data, iteration, True)

    def _forward_step(
        self, data, iteration, is_eval):  # pylint: disable=unused-argument
        outputs = self.generate(data, is_eval)

        if outputs is not None:
            rets = self.decode_internal(outputs, data)
            return rets

    def forward_step(self, data, iteration=0):

        rets = self._forward_step(data, iteration, False)
        rets["uid"] = data["uid"]
        # collect metric
        response_token_length = rets["response_token_length"]
        prompt_token_length = rets["prompt_token_length"]
        seq_len = [
            l1 + l2 for l1, l2 in zip(prompt_token_length, response_token_length)
        ]
        clip_ratio = sum(
            l >= self.module_args.get("seq_length") for l in seq_len
        ) / len(seq_len)
        inference_stats = {
            "response_token_length": sum(response_token_length)
            / len(response_token_length),
            "prompt_token_length": sum(prompt_token_length) / len(prompt_token_length),
            "response_clip_ratio": clip_ratio,
        }
        self._metric_list.append(inference_stats)
        return rets

    def decode_internal(
        self, batched_outputs: List[Dict], input_data
    ):
        data_source_list = input_data["data_source"]
        ground_truth_list = input_data["ground_truth"]
        max_tokens_length = self.module_args.get("seq_length")
        all_tokens = []
        str_outputs = []
        prompt_token_ids: list = input_data["input_ids"]
        prompt_token_length = []
        response_token_length = []
        data_sources = []
        ground_truths = []
        for idx, output in enumerate(batched_outputs):
            data_source = data_source_list[idx] if data_source_list else ""
            data_sources.append(data_source)

            ground_truth = ground_truth_list[idx] if ground_truth_list else ""
            ground_truths.append(ground_truth)
            prompt_token = input_data['input_ids'][idx]
            output_tokens = output['output_ids']

            response_token_length.append(output['meta_info']['completion_tokens'])
            prompt_token_length.append(output['meta_info']['prompt_tokens'])
            str_outputs.append(
                self.tokenizer.tokenizer.decode(
                    output_tokens, skip_special_tokens=True
                )
            )
            all_tokens.append(torch.tensor(prompt_token + output_tokens))

        all_tokens = [
            F.pad(
                all_token,
                (0, max_tokens_length - all_token.shape[0]),
                value=self.tokenizer.tokenizer.pad_token_id,  # just pad_token_id
            )
            for all_token in all_tokens
        ]
        all_tokens = torch.vstack(all_tokens)
        print("str_outputs", str_outputs[0])
        print("data_sources", data_sources[0])
        print("ground_truth", ground_truths[0])

        return {
            "all_tokens": all_tokens,
            "str_outputs": str_outputs,
            "prompt_token_ids": prompt_token_ids,
            "prompt_token_length": prompt_token_length,
            "response_token_length": response_token_length,
            "data_source": data_sources,
            "ground_truth": ground_truths,
        }
