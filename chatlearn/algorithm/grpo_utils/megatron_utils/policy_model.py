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

from megatron.core import mpu
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.training import get_args
from torch import Tensor

from chatlearn.configs.base import BaseModelConfig

from ..loss_gallery import calculate_grpo_loss, calculate_gspo_loss
from .train_helper import entropy_from_tensor_parallel_logits, reduce_from_context_parallel_region


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

        forward_logprob = reduce_from_context_parallel_region(forward_logprob, self.module_args.packing, training_inputs)

        old_logprobs = training_inputs["old_logprobs"]
        ref_logprobs = training_inputs["ref_logprobs"]
        advantages = training_inputs["advantages"]


        if self.module_args.use_group_sequence_policy:
            (
                pg_loss, 
                is_positive_clipped, 
                is_negative_clipped,
                is_clipped,
            ) = calculate_gspo_loss(
                log_probs=forward_logprob,
                old_log_probs=old_logprobs,
                advantages=advantages,
                diff_clip_ratio=self.module_args.diff_clip_ratio,
                pos_clip_ratio=self.module_args.pos_clip_ratio,
                neg_clip_ratio=self.module_args.neg_clip_ratio,
                final_clip_ratio=self.module_args.final_clip_ratio,
                loss_mask = training_inputs['all_token_loss_mask']
            )
        else:
            pg_loss = calculate_grpo_loss(
                log_probs=forward_logprob,
                old_log_probs=old_logprobs,
                advantages=advantages,
                diff_clip_ratio=self.module_args.diff_clip_ratio,
                pos_clip_ratio=self.module_args.pos_clip_ratio,
                neg_clip_ratio=self.module_args.neg_clip_ratio,
                final_clip_ratio=self.module_args.final_clip_ratio
            )

        entropy_loss = entropy_from_tensor_parallel_logits(all_token_logits).permute(1, 0)

        entropy_loss = reduce_from_context_parallel_region(entropy_loss, self.module_args.packing, training_inputs)

        kl = ref_logprobs - forward_logprob
        ratio = torch.exp(kl)
        ratio[~training_inputs['all_token_loss_mask'].bool()] = 1
        assert not torch.isinf(ratio).any(), "kl loss ratio has inf values"
        assert not torch.isnan(ratio).any(), "kl loss ratio has nan values"
        kld = (ratio - kl - 1).contiguous()
        kl_loss = torch.clamp(kld, min=-10, max=10)

        if self.module_args.use_group_sequence_policy:
            return {
                'pg_loss': pg_loss,
                'entropy_loss': entropy_loss,
                'kl_loss': kl_loss,
                'is_positive_clipped': is_positive_clipped,
                'is_negative_clipped': is_negative_clipped,
                'is_clipped': is_clipped,
                'is_positive_clipped_sample_average': is_positive_clipped,
                'is_negative_clipped_sample_average': is_negative_clipped,
                'is_clipped_sample_average': is_clipped,
            }
        else:
            return {
                'pg_loss': pg_loss,
                'entropy_loss': entropy_loss,
                'kl_loss': kl_loss
            }