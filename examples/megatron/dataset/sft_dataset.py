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
"""sft dataset"""

import json

import numpy as np
import torch
from megatron import get_tokenizer
from megatron import print_rank_0


def build_train_valid_test_datasets(data_prefix, seq_length):
    """Build train, valid, and test datasets."""

    print_rank_0("Separate data paths provided for train, valid & test. Split string will be ignored.")

    train_dataset_path, valid_dataset_path, test_dataset_path = data_prefix

    # Single dataset.
    if train_dataset_path is not None:
        train_dataset = SFTDataset(train_dataset_path, seq_length)

    if valid_dataset_path is not None:
        valid_dataset = SFTDataset(valid_dataset_path, seq_length)

    if test_dataset_path is not None:
        test_dataset = SFTDataset(test_dataset_path, seq_length)

    return train_dataset, valid_dataset, test_dataset


class SFTDataset(torch.utils.data.Dataset):
    """
    SFT Dataset
    """

    def __init__(self, data_path, max_seq_length):
        self.data_path = data_path
        self.max_seq_length = max_seq_length + 1
        with open(self.data_path, 'r', encoding="utf-8") as f:
            self.dataset = f.readlines()
            self.dataset = [json.loads(item) for item in self.dataset]
        self.tokenizer = get_tokenizer()
        self.pad_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else self.tokenizer.pad_id
        self.bos_id = self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eod

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        query = self.dataset[idx]['prompt']
        response = self.dataset[idx]['response']
        query_toks, all_toks = [
            self.tokenizer.tokenize(
                text
            )
            for text in [query, query + response]
        ]
        all_toks = all_toks + [self.tokenizer.eod]

        if len(all_toks) >= self.max_seq_length:
            if len(query_toks) >= self.max_seq_length and len(all_toks) - len(query_toks) >= self.max_seq_length:
                query_toks, all_toks = query_toks[-self.max_seq_length // 2:], \
                                       all_toks[len(query_toks) - self.max_seq_length // 2:len(
                                           query_toks) + self.max_seq_length // 2]
            elif len(all_toks) - len(query_toks) >= self.max_seq_length:
                all_toks = all_toks[:self.max_seq_length]
            else:
                query_toks, all_toks = query_toks[-(self.max_seq_length - len(all_toks) + len(query_toks)):], \
                                       all_toks[-self.max_seq_length:]

        query_toks_len, all_toks_len = len(query_toks), len(all_toks)

        if all_toks_len < self.max_seq_length:
            all_toks = all_toks + [self.pad_id] * (self.max_seq_length - all_toks_len)

        label_mask = [1] * len(all_toks)
        label_mask = np.array(label_mask, dtype=np.int64)
        all_toks = np.array(all_toks, dtype=np.int64)

        label_mask[:query_toks_len] = 0
        label_mask[all_toks_len:] = 0

        return {"text": np.array(all_toks, dtype=np.int64), "loss_mask": np.array(label_mask, dtype=np.int64)}
