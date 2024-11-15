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

import copy
import random

import torch
import torch.nn.functional as F

# from chatlearn import VLLMModule
from chatlearn.models.vllm_module2 import VLLMModule2 as VLLMModule
from examples.megatron.data.prompt_dataset import VLLMPromptPipeline

from .utils import get_loss_mask


class VLLMPolicyInference(VLLMModule):
    """Policy vLLM Inference"""

    def build_dataset(self, train_prompts, is_eval=False):
        if is_eval:
            duplicated_train_prompts = train_prompts
        else:
            if self.model_args["init_shuffle_prompts"] == 2:
                # this is to approximate n epochs and by pass the chatlearn epoch which currently hangs
                # append epochs and shuffle epoch by epoch and attach them together
                # and now num_inference_per_prompt is number of epochs
                duplicated_train_prompts = []
                for i in range(self.model_args["num_inference_per_prompt"]):
                    train_prompts_cp = copy.deepcopy(train_prompts)
                    random.shuffle(train_prompts_cp)
                    duplicated_train_prompts.extend(train_prompts_cp)
            elif self.model_args["init_shuffle_prompts"] == 0:
                # otherwise, it's a huge epoch
                duplicated_train_prompts = []
                for p in train_prompts:
                    duplicated_train_prompts.extend([p for i in range(self.model_args["num_inference_per_prompt"])])
            else:
                raise Exception(f"unsupported init_shuffle_prompts {self.model_args['init_shuffle_prompts']}, expect 0 or 2.")

        max_prompt_length = (
            self.model_args.get("seq_length") - self.model_args.get("max_new_tokens")
        )
        prompt_key = self.model_args.get("prompt_key")
        num_tokenize_threads = self.model_args.get("num_tokenize_threads")
        # TODO: read from files
        prompts_dataset = VLLMPromptPipeline(
            duplicated_train_prompts, max_prompt_length, self.tokenizer.tokenizer, prompt_key, num_tokenize_threads)

        return prompts_dataset

    def eval_forward(self, data, iteration=0):
        return self._forward_step(data, iteration, True)

    def _forward_step(self, data, iteration, is_eval): # pylint: disable=unused-argument
        outputs = self.generate_vllm(data, is_eval)
        rets = self.decode_internal(outputs)
        return rets

    def forward_step(self, data, iteration=0):
        return self._forward_step(data, iteration, False)

    def decode_internal(self, batched_outputs):
        max_tokens_length = self.model_args.get("seq_length")
        no_padded_query_ids = []
        all_tokens = []
        str_outputs = []
        str_prompts = []
        logprobs = []
        for output in batched_outputs:
            num_responses_per_prompt = len(output.outputs)
            for res_idx in range(num_responses_per_prompt):
                str_prompts.append(output.prompt)
                str_outputs.append(output.outputs[res_idx].text)
                no_padded_query_ids.append(torch.tensor(output.prompt_token_ids))

                output_logprobs = []
                for idx, probs in enumerate(output.outputs[res_idx].logprobs):
                    prob = probs[output.outputs[res_idx].token_ids[idx]]
                    if isinstance(prob, float):
                        output_logprobs.append(prob)
                    else:
                        output_logprobs.append(prob.logprob)
                logprob = torch.tensor(output_logprobs)
                if output.prompt_logprobs is not None:
                    prompt_logprobs = []
                    for idx, prompt_token_id in enumerate(output.prompt_token_ids):
                        if idx == 0:
                            continue
                        prompt_logprobs.append(output.prompt_logprobs[idx][prompt_token_id])
                else:
                    prompt_logprobs = [0.0 for _ in range(len(output.prompt_token_ids) - 1)]
                output_tokens = list(output.outputs[res_idx].token_ids)
                all_tokens.append(torch.tensor(output.prompt_token_ids + output_tokens))
                prompt_logprobs = torch.tensor(prompt_logprobs)
                logprob = torch.cat([prompt_logprobs, logprob])
                logprobs.append(logprob)

        all_tokens = [
            F.pad(
                all_token,
                (0, max_tokens_length - all_token.shape[0]),
                value=self.tokenizer.tokenizer.eos_token_id,  # just pad_token_id
            )
            for all_token in all_tokens
        ]
        all_tokens = torch.vstack(all_tokens)

        logprobs = [
            F.pad(
                logprob,
                (0, max_tokens_length - logprob.shape[0] - 1),
                value=0.0
            )
            for logprob in logprobs
        ]
        logprobs = torch.vstack(logprobs)

        prompt_sizes = torch.tensor([len(q) for q in no_padded_query_ids], device=all_tokens.device)
        loss_mask = get_loss_mask(all_tokens, self.tokenizer.tokenizer.eos_token_id, prompt_sizes)
        loss_mask = loss_mask.to("cpu")
        return {"all_tokens": all_tokens, "str_outputs": str_outputs, "str_prompts": str_prompts,
            "no_padded_query_ids": no_padded_query_ids, "logprobs": logprobs,
            "loss_mask": loss_mask}
