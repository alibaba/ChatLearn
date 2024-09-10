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
"""finetune sft"""

from functools import partial

import torch
from megatron.training import get_args
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.training import print_rank_0
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.legacy.model import GPTModel as LegacyGPTModel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.training import pretrain
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module

from examples.megatron.data.sft_dataset import build_train_valid_test_datasets
from examples.megatron.models.utils import has_config_in_args


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    args = get_args()

    if args.use_legacy_models:
        if has_config_in_args(LegacyGPTModel):
            from megatron.training.arguments import core_transformer_config_from_args # pylint: disable=import-outside-toplevel

            config = core_transformer_config_from_args(args)
            model = LegacyGPTModel(
                config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
        else:
            model = LegacyGPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    else:
        # MCore models consistantly have `config`
        from megatron.training.arguments import core_transformer_config_from_args # pylint: disable=import-outside-toplevel

        config = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.transformer_impl == "transformer_engine":
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

        model = MCoreGPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base
        )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["text", "loss_mask"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    loss_mask = data_b["loss_mask"].long()[:, 1:].contiguous()

    # Get the masks and position ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples): # pylint: disable=unused-argument
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for LLaMA ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        seq_length=args.seq_length)
    print_rank_0("> finished creating LLaMA datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step)
