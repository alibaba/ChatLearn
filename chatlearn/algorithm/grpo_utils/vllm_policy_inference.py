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
from typing import Dict, List, Any
from collections import deque, defaultdict
from collections import Counter
import time

import copy

import torch
import torch.nn.functional as F

from chatlearn.data.prompt_dataset import PromptPipeline
from chatlearn.runtime.decorator import timeit, compute_decorator
# pylint: disable=ungrouped-imports
from chatlearn.models.vllm_module import VLLMModule


class VLLMPolicyInference(VLLMModule):
    """Policy vLLM Inference"""

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
        self, data, iteration, is_eval
    ):  # pylint: disable=unused-argument
        outputs = self.generate_vllm(data, is_eval, iteration=iteration)

        if outputs is not None:
            rets = self.decode_internal(outputs, data)
            return rets
    
    @timeit("vllm_forward_step")
    @compute_decorator(trainable=False, rollout=True)
    def forward_step(self, data: List[Dict[str, Any]], iteration=0, **kwargs) -> List[Dict[str, Any]]: # pylint: disable=unused-argument
        round_track = {}
        if "rollout_round" in data[0]:
            round_track = dict(Counter(d["rollout_round"] for d in data))
        start_time = time.time()
        rets = self._forward_step(data, iteration, False)
        # collect metric
        seq_len = self.module_args.get("seq_length")
        response_token_length = [ret["response_token_length"] for ret in rets]
        prompt_token_length = [ret["prompt_token_length"] for ret in rets]
        clip_ratio = sum(
            ret["response_token_length"] >= ret.get("max_generate_token_length", seq_len) for ret in rets
        ) / len(rets)
        inference_stats = {
            "response_token_length": sum(response_token_length)
            / len(response_token_length),
            "prompt_token_length": sum(prompt_token_length) / len(prompt_token_length),
            "response_clip_ratio": clip_ratio,
        }
        round_track.update({'time': time.time() - start_time})
        print(f"debugyy round_track for each rollout actor {round_track}")
        self._metric_list.append(inference_stats)
        return rets

    def decode_internal(
        self, outputs_list: List[Dict[str, Any]], input_data_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        data_output = []
        for output, input_data in zip(outputs_list, input_data_list):
            num_responses_per_prompt = len(output.outputs)
            for res_idx in range(num_responses_per_prompt):
                data_obj = copy.deepcopy(input_data)
                prompt_token_ids = output.prompt_token_ids
                output_tokens = list(output.outputs[res_idx].token_ids)
                response_token_length = len(output_tokens)
                prompt_token_length = len(output.prompt_token_ids)
                str_outputs = self.tokenizer.tokenizer.decode(
                        output_tokens, skip_special_tokens=True
                    )
                all_tokens = torch.tensor(output.prompt_token_ids + output_tokens)
                data_obj.update(
                    {
                        "all_tokens": all_tokens,
                        "response_token_length": response_token_length,
                        "str_outputs": str_outputs,
                    }
                )
                if "rollout_round" in data_obj:
                    data_obj["rollout_round"] += 1
                data_output.append(data_obj)
        print("str_outputs", data_output[0]["str_outputs"])
        print("data_sources", data_output[0]["data_source"])
        print("ground_truth", data_output[0]["ground_truth"])

        return data_output