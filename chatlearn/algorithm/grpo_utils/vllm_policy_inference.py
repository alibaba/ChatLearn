# pylint: skip-file
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
"""vllm policy inference"""
from typing import Dict, List

import torch
import torch.nn.functional as F

from chatlearn.data.prompt_dataset import PromptPipeline
# pylint: disable=ungrouped-imports
from chatlearn.models.vllm_module import VLLMModule


class VLLMPolicyInference(VLLMModule):
    """Policy vLLM Inference"""

    def build_dataset(self, prompts: List[Dict], is_eval=False):
        # prompts seems like the total data set by engine.set_dataset(dataset)
        seq_length = self.module_args.get("seq_length")
        assert len(prompts)>0, 'Dataset is empty'

        if 'images' in prompts[0].keys():
            from chatlearn.data.vl_prompt_dataset import PromptPipeline
            prompts_dataset = PromptPipeline(
                prompts,
                seq_length,
                self.tokenizer.tokenizer,
                self.processor,
                enable_thinking=self.module_args.get("enable_thinking", False),
            )
        else:
            from chatlearn.data.prompt_dataset import PromptPipeline
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
        self, data, iteration, is_eval
    ):  # pylint: disable=unused-argument
        outputs = self.generate_vllm(data, is_eval, iteration=iteration)
        # for get rule reward function
        data_source_list = data["data_source"]
        ground_truth_list = data["ground_truth"]
        prompt_position_ids_list = data.get("position_ids", None)
        rope_deltas_list = data.get("rope_deltas", None)
        pixel_value_list = data.get("pixel_values", None)
        image_grid_thw_list = data.get("image_grid_thw", None)

        if outputs is not None:
            rets = self.decode_internal(outputs, data_source_list, ground_truth_list, prompt_position_ids_list, rope_deltas_list, pixel_value_list, image_grid_thw_list)
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
            1 for l in seq_len if l >= self.module_args.get("seq_length")
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
        self, batched_outputs, data_source_list=None, ground_truth_list=None, prompt_position_ids_list=None, rope_deltas_list=None, pixel_value_list=None, image_grid_thw_list=None
    ):
        max_tokens_length = self.module_args.get("seq_length")
        all_tokens = []
        str_outputs = []
        prompt_token_ids: list = []
        prompt_token_length = []
        response_token_length = []
        data_sources = []
        ground_truths = []
        prompts_position_ids = []
        prompts_rope_deltas = []

        for idx, output in enumerate(batched_outputs):
            num_responses_per_prompt = len(output.outputs)
            data_source = data_source_list[idx] if data_source_list else ""
            ground_truth = ground_truth_list[idx] if ground_truth_list else ""
            prompt_position_ids = prompt_position_ids_list[idx] if prompt_position_ids_list else []
            prompt_rope_deltas = rope_deltas_list[idx] if rope_deltas_list else []

            for res_idx in range(num_responses_per_prompt):
                # str_prompts.append(output.prompt)
                prompt_token_ids.append(output.prompt_token_ids)
                output_tokens = list(output.outputs[res_idx].token_ids)
                response_token_length.append(len(output_tokens))
                prompt_token_length.append(len(output.prompt_token_ids))
                str_outputs.append(
                    self.tokenizer.tokenizer.decode(
                        output_tokens, skip_special_tokens=True
                    )
                )
                data_sources.append(data_source)
                ground_truths.append(ground_truth)
                prompts_position_ids.append(prompt_position_ids)
                prompts_rope_deltas.append(prompt_rope_deltas)
                all_tokens.append(torch.tensor(output.prompt_token_ids + output_tokens))

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
            "prompt_position_ids": prompts_position_ids,
            "prompt_rope_deltas": prompts_rope_deltas,
            "pixel_values": pixel_value_list,
            "image_grid_thw": image_grid_thw_list
        }
