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

from typing import Literal, Optional

import torch
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

class PolicyModel(GPTModel):
    """PolicyModel"""

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "mrope", "none"
        ] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        module_args: Optional[BaseModelConfig] = None
    ) -> None:
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
        )

        # chatlearn module args
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
    ):

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

        # [b s h] => [s b h]
        all_token_logits = hidden_states_or_logits.transpose(0, 1).contiguous()
        old_logprobs = training_inputs["old_logprobs"]
        ref_logprobs = training_inputs["ref_logprobs"]
        advantages = training_inputs["advantages"]

        forward_logprob = (
            self.compute_language_model_loss(labels, all_token_logits) * -1
        )

        pg_loss = calculate_grpo_loss(
            log_probs=forward_logprob,
            old_log_probs=old_logprobs,
            advantages=advantages,
            diff_clip_ratio=self.module_args.diff_clip_ratio,
            pos_clip_ratio=self.module_args.pos_clip_ratio,
            neg_clip_ratio=self.module_args.neg_clip_ratio,
            final_clip_ratio=self.module_args.final_clip_ratio,
        )

        entropy_loss = entropy_from_tensor_parallel_logits(all_token_logits).transpose(0, 1)

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
