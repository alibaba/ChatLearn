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

from typing import Literal, Optional, Dict, Any, Union

import torch

from flash_attn.bert_padding import pad_input

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args
from torch import Tensor

from chatlearn.configs.common import BaseModelConfig

from ..loss_gallery import calculate_grpo_loss
from .train_helper import entropy_from_tensor_parallel_logits


# TODO: replace this class with GPTModel
class PolicyModel(GPTModel):
    """PolicyModel"""

    def __init__(self, *args, module_args: Optional[BaseModelConfig] = None, **kwargs):
        """Create a Megatron-Core Policy Model. For more descriptions, please
        refer to `megatron.core.models.gpt.GPTModel`

        Args:
            module_args (Optional[BaseModelConfig], optional): Arguments for chatlearn modules.
            Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.module_args = module_args

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        training_inputs: dict = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # untransposed hidden_states or transposed logits with shape [b, s, h]
        hidden_states_or_logits = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=None,
            loss_mask=loss_mask,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
        )

        if not self.post_process:
            return hidden_states_or_logits

        if training_inputs is None:
            return (
                self.compute_language_model_loss(
                    labels,
                    hidden_states_or_logits.transpose(
                        0, 1
                    ).contiguous(),  # [b s h] => [s b h]
                )
                if labels is not None
                else hidden_states_or_logits
            )

        return self._compute_all_losses(
            all_token_logits=hidden_states_or_logits.transpose(0, 1).contiguous(),
            labels=labels,
            training_inputs=training_inputs
        )

    def _compute_all_losses(
        self, 
        all_token_logits: torch.Tensor, 
        labels: torch.Tensor, 
        training_inputs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Compute all required losses.
        
        Args:
            all_token_logits (torch.Tensor): logits of input tokens. shape: [s, b, h] or [total_nnz, 1, h]
            labels (torch.Tensor): labels of input tokens. shape: [s, b] or [total_nnz, 1]
            training_inputs (Dict[str, Any]): All training inputs.
        
        """
        forward_logprob = (
            self.compute_language_model_loss(labels, all_token_logits) * -1
        )

        # NOTE: before loss computation, we need to unpack the packed inputs
        if self.module_args.packing:
            forward_logprob = pad_input(
                forward_logprob[0, :forward_logprob.shape[1] - training_inputs['pad_size']].unsqueeze(-1), 
                training_inputs['indices'], 
                training_inputs['ori_batch_size'], 
                training_inputs['ori_seq_len']
            ).squeeze(-1)

        old_logprobs = training_inputs["old_logprobs"]
        ref_logprobs = training_inputs["ref_logprobs"]
        advantages = training_inputs["advantages"]


        pg_loss = calculate_grpo_loss(
            log_probs=forward_logprob,
            old_log_probs=old_logprobs,
            advantages=advantages,
            diff_clip_ratio=self.module_args.diff_clip_ratio,
            pos_clip_ratio=self.module_args.pos_clip_ratio,
            neg_clip_ratio=self.module_args.neg_clip_ratio,
            final_clip_ratio=self.module_args.final_clip_ratio,
        )

        entropy_loss = entropy_from_tensor_parallel_logits(all_token_logits)
        if self.module_args.packing:
            entropy_loss = pad_input(
                entropy_loss[:entropy_loss.shape[0] - training_inputs['pad_size']], 
                training_inputs['indices'], 
                training_inputs['ori_batch_size'], 
                training_inputs['ori_seq_len']
            ).squeeze(-1)
        else:
            entropy_loss = entropy_loss.transpose(0, 1)

        kl = ref_logprobs - forward_logprob
        ratio = torch.exp(kl)
        assert not torch.isinf(ratio).any(), "kl loss ratio has inf values"
        assert not torch.isnan(ratio).any(), "kl loss ratio has nan values"
        kld = (ratio - kl - 1).contiguous()
        kl_loss = torch.clamp(kld, min=-10, max=10)

        return {
            'pg_loss': pg_loss,
            'entropy_loss': entropy_loss,
            'kl_loss': kl_loss,
        }
