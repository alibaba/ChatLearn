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
"""finetune reward"""

from functools import partial

import torch
from dataset.reward_dataset import build_train_valid_test_datasets_for_rm
from megatron import get_args
from megatron import get_timers
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.core import parallel_state
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import schedules
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import get_ltor_masks_and_position_ids
from models.reward_model import model_provider


def get_tensor_shapes_reward( # pylint: disable=unused-argument
    *,
    rank: int,
    model_type: ModelType,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
):
    tensor_shapes = []

    if config.sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
    args = get_args()
    if args.max_response > 1:
        tensor_shapes.append((seq_length, args.max_response*micro_batch_size, config.hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes
schedules.get_tensor_shapes = get_tensor_shapes_reward

def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text', "all_length", "num_responses", "all_score"]
    datatype = torch.int64
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    seq_len = data_b['text'].shape[-1]
    tokens = data_b['text'].reshape(-1, seq_len)

    all_length = data_b["all_length"]
    num_responses = data_b["num_responses"]
    all_score = data_b["all_score"]

    eos_indices = torch.clamp(torch.concat([all_length[i][:num_responses[i]] for i in range(len(num_responses))]),
                              min=1, max=tokens.shape[-1]) - 1

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, eos_indices, num_responses, attention_mask, position_ids, all_score


def loss_func(output_tensor, num_responses, all_score):
    '''
    output_tensor: (B*num response) x score_dim
    num_respones: B
    all_score: B x (max num reponse) x score_dim
    '''

    idx = 0
    loss = 0
    compare_count_all = 0
    true_count_all = 0
    valid_sample = 0
    all_score = all_score.unsqueeze(-1)

    for i, n in enumerate(num_responses):
        gt_scores = all_score[i][:n].unsqueeze(0).expand(n, -1, -1)  # num response x num response x A
        loss_mask = (gt_scores - gt_scores.transpose(1, 0)) > 0
        valid_compare_num = torch.sum(loss_mask)
        if valid_compare_num == 0:
            loss_mat = output_tensor[idx: idx + n].unsqueeze(0).expand(n, -1, -1)
            loss_mat = (loss_mat - loss_mat.transpose(1, 0))
            loss -= torch.sum(torch.log(torch.sigmoid(loss_mat)) * loss_mask)
            true_count_all += torch.sum(loss_mat * loss_mask > 0)
            idx += n
            compare_count_all += valid_compare_num

        else:
            loss_mat = output_tensor[idx: idx + n].unsqueeze(0).expand(n, -1, -1)
            loss_mat = (loss_mat - loss_mat.transpose(1, 0))
            loss -= torch.sum(torch.log(torch.sigmoid(loss_mat)) * loss_mask) / valid_compare_num

            true_count_all += torch.sum(loss_mat * loss_mask > 0)
            idx += n
            compare_count_all += valid_compare_num
            valid_sample += 1
    loss = loss / (valid_sample + 1e-5)
    true_count_all = true_count_all.detach() / (compare_count_all + 1e-5) * 100

    averaged_loss = average_losses_across_data_parallel_group([loss])

    true_count_all = average_losses_across_data_parallel_group([true_count_all])
    return loss, {'lm loss': averaged_loss[0], 'accuracy': true_count_all[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, eos_token_position, num_responses, attention_mask, position_ids, all_score = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, pooling_sequence_index=eos_token_position)

    return output_tensor, partial(loss_func, num_responses=num_responses, all_score=all_score)


def train_valid_test_datasets_provider(train_val_test_num_samples): # pylint: disable=unused-argument
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets_for_rm(
        input_file=args.data_path,
        tokenizer=get_tokenizer(),
        max_length=args.seq_length
    )
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def add_extra_args(parser):
    group = parser.add_argument_group(title='reward training')
    group.add_argument('--max-response', type=int, default=2,
                    help='number of response for each query, only used for reward model training, default set by 2')
    group.add_argument('--select-max-response', type=str, default='firstk',
                    help='if response number exceed max-response, how to select response to train.')
    return parser


if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=add_extra_args
    )
