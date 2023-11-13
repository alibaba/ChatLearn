# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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

import torch
import torch.nn.functional as F

from dataset.prompt_dataset import VLLMPromptPipeline
from models.vllm_policy_model import VLLMPolicyModel

from vllm.transformers_utils.tokenizer import get_tokenizer

from chatlearn import RLHFVLLMModule
from chatlearn.utils.vllm_utils import get_model, print_rank_0
from .utils import get_loss_mask


class VLLMPolicyInference(RLHFVLLMModule):
    """Policy vLLM Inference"""

    def setup(self):
        # Set up model and load checkpoint
        self.tokenizer = get_tokenizer(
            self.model_args.get("tokenizer"),
            tokenizer_mode="auto",
            trust_remote_code=False,
            tokenizer_revision=None,
            revision=None
        )
        model = [get_model(self.model_provider, self.model_args, wrap_with_ddp=False)]

        assert len(model) == 1, "Above condition should have caught this"
        self.model = model[0]

        return 'ok'

    def build_dataset(self, train_prompts):
        '''
        framework source: dataset = self.build_dataset(data)
        :param train_prompts: all train prompts used in this training run??
        :return:
            a torch.utils.data.Dataset object for prompts_loader of all prompts, and
        '''
        max_prompt_length = (
            self.model_args.get("seq_length") - self.model_args.get("max_new_tokens")
        )
        # TODO: read from files
        prompts_dataset = VLLMPromptPipeline(
            train_prompts, max_prompt_length, self.tokenizer)

        return prompts_dataset

    def model_provider(self):
        """Build the model."""
        print_rank_0('building vLLM model ...')
        model = VLLMPolicyModel(self.model_config, self.model_args)

        return model

    def eval_forward(self, data):
        return self._forward_step(data, 0, eval_mode=True)

    def _forward_step(self, data, iteration, eval_mode: bool):
        '''
        RLHF calling
        rlhf framework source:     policy_output = self.policy.forward_step(query)
        :param data: entire global batch?? micro_batch?
        :return:
            data using current microbatch
            {"all_tokens": tokens,  "str_samples": str_samples,
                "str_prompts": str_prompts, "str_outputs": str_outputs, "logprobs": all_log_probs,
                "no_padded_query_ids": no_padded_query_ids}
        '''
        assert iteration >= 0
        assert eval_mode, "Expect eval mode is True for vllm policy model."
        return self.model(
            data["input_ids"],
            data["positions"],
            kv_caches=data["kv_caches"],
            input_metadata=data["input_metadata"],
            cache_events=data["cache_events"]
        )

    def _add_request(self, data):
        return self._add_request_internal(data["prompt"], data["input_ids"])

    def forward_step(self, data, iteration=0): # pylint: disable=unused-argumen
        seq_group_metadata_list = data["seq_group_metadata_list"]
        blocks_to_swap_in = data["blocks_to_swap_in"]
        blocks_to_swap_out = data["blocks_to_swap_out"]
        blocks_to_copy = data["blocks_to_copy"]

        outputs = self.execute_step(
            seq_group_metadata_list, blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        return outputs

    def decode_internal(self, batched_outputs):
        '''
        RLHF calling
        rlhf framework source:     policy_output = self.policy.forward_step(query)
        :param batched_outputs: batched_outputs
        :return:
            data using current microbatch
            {"all_tokens": tokens,  "str_samples": str_samples,
                "str_prompts": str_prompts, "str_outputs": str_outputs, "logprobs": all_log_probs,
                "no_padded_query_ids": no_padded_query_ids}
        '''
        no_padded_query_ids = []
        outputs_tokens = []
        str_outputs = []
        str_prompts = []
        str_samples = []
        logprobs = []
        max_prompt_len = 0
        max_new_tokens = 0
        for output in batched_outputs:
            max_prompt_len = max(max_prompt_len, len(output.prompt_token_ids))
            str_prompts.append(output.prompt)
            str_outputs.append(output.outputs[0].text)
            str_samples.append(str_prompts[-1] + str_outputs[-1])
            no_padded_query_ids.append(torch.tensor(output.prompt_token_ids))
            max_new_tokens = max(max_new_tokens, len(output.outputs[0].token_ids))
            outputs_tokens.append(torch.tensor(output.outputs[0].token_ids))
            logprobs.append(torch.tensor([probs[output.outputs[0].token_ids[idx]] for idx, probs in enumerate(output.outputs[0].logprobs)]))

        prompts_tokens = [
            F.pad(
                prompts_token,
                (0, max_prompt_len - prompts_token.shape[0]),
                value=self.tokenizer.eos_token_id,  # just pad_token_id
            )
            for prompts_token in no_padded_query_ids
        ]
        prompts_tokens_tensor = torch.vstack(prompts_tokens).to(torch.cuda.current_device())

        outputs_tokens = [
            F.pad(
                output_token,
                (0, max_new_tokens - output_token.shape[0]),
                value=self.tokenizer.eos_token_id,  # just pad_token_id
            )
            for output_token in outputs_tokens
        ]
        output_tokens_tensor = torch.vstack(outputs_tokens).to(torch.cuda.current_device())

        logprobs = [
            F.pad(
                logprob,
                (0, max_new_tokens - logprob.shape[0]),
                value=0.0
            )
            for logprob in logprobs
        ]
        logprobs = torch.vstack(logprobs).to(torch.cuda.current_device())
        logprobs_left_padding = torch.zeros(
            [logprobs.size(0), logprobs.size(1) - 1], dtype=logprobs.dtype, layout=logprobs.layout, device=logprobs.device)
        logprobs = torch.cat([logprobs_left_padding, logprobs], dim=1)

        all_tokens = torch.cat([prompts_tokens_tensor, output_tokens_tensor], dim=1)

        prompt_sizes = torch.tensor([len(q) for q in no_padded_query_ids], device=all_tokens.device)
        loss_mask = get_loss_mask(all_tokens, self.tokenizer.eos_token_id, prompt_sizes)

        return {"all_tokens": all_tokens, "str_outputs": str_outputs, "str_prompts": str_prompts,
            "no_padded_query_ids": no_padded_query_ids, "logprobs": logprobs,
            "loss_mask": loss_mask}
