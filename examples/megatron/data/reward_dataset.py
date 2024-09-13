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
"""reward dataset"""

import json
import random

import numpy as np
import torch
from megatron.training import get_args
from torch.utils.data import Dataset


def open_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    opt = []
    for line in lines:
        opt.append(json.loads(line))
    return opt


class RewardDataset(Dataset):
    """Torch dataset"""

    def __init__(self, input_file, tokenizer, max_length=2048):
        self.input_file = input_file
        self.dataset = open_jsonl(self.input_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = self.tokenizer.eod

        args = get_args()
        self.max_response = args.max_response
        self.select_max_response = args.select_max_response

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        query = self.tokenizer.tokenize(data['query'])
        responses = []

        if len(data['response']) > self.max_response:
            if self.select_max_response == 'firstk':
                data['response'] = data['response'][0:self.max_response]
                data['score'] = data['score'][0:self.max_response]
            elif self.select_max_response == 'topk':
                scores = np.array(data['score'])
                idx = scores.argsort()[-self.max_response:][::-1]
                data['response'] = [data['response'][id] for id in idx]
                data['score'] = [data['score'][id] for id in idx]
            elif self.select_max_response == 'random':
                idx = random.sample(list(range(len(data['response']))), self.max_response)
                data['response'] = [data['response'][id] for id in idx]
                data['score'] = [data['score'][id] for id in idx]
            else:
                raise RuntimeError(f"non-supported select_max_response {self.select_max_response}")

        for res in data['response']:
            tok_res = self.tokenizer.tokenize(res) + [self.tokenizer.eod]
            responses.append(tok_res)

        tokenized_item = [data['score'], query]
        tokenized_item.extend(responses)

        return self.extract_feature_from_document_rewardmodel(tokenized_item, self.tokenizer, self.max_length)

    def extract_feature_from_document_rewardmodel(self, document_piece, tokenizer, max_length):
        """Creates `TrainingInstance`s for a single document."""
        num2char = None  # tokenizer.num2char
        assert len(document_piece[0]) == len(document_piece[2:])
        selected_sample = [(sco, repson) for sco, repson in zip(document_piece[0], document_piece[2:])] # pylint: disable=unnecessary-comprehension
        scores = [item[0] for item in selected_sample]
        response_piece = [item[1] for item in selected_sample]

        query_piece = [document_piece[1]]
        document_piece = query_piece + response_piece

        if num2char is not None:
            document_piece_ = []
            for segment in document_piece:
                segment_ = []
                for token in segment:
                    if token in num2char:
                        segment_.extend(num2char[token])
                    else:
                        segment_.append(token)
                document_piece_.append(segment_)
            document_piece = document_piece_
        lens = len(document_piece)
        assert lens >= 2, f"document_piece len should be >=2, but got {lens}"

        out_sequences = []
        for i in range(lens - 1):
            out_sequences.append(preprocess(document_piece[0], document_piece[i + 1], max_length, tokenizer))
        outs = batchify_to_max_length({'data': out_sequences, 'num_response': len(out_sequences), 'scores': scores})

        return outs


def preprocess(prompt, response, max_length, tokenizer):
    prompt_chunk = []
    completion_chunk = []

    prompt_chunk.extend(prompt)
    completion_chunk.extend(response)
    # TODO: need append it to be consistent with trained models, we may need remove later
    completion_chunk.append(tokenizer.eod)

    # output prompt length
    prompt_length = len(prompt_chunk)
    completion_length = len(completion_chunk)
    if prompt_length > max_length and completion_length > max_length:
        # When both the prompt and completion exceed the maximum length,
        # try to keep the most important parts of both and center them.
        prompt_chunk = prompt_chunk[-(max_length // 2):]
        completion_chunk = completion_chunk[:max_length // 2 + 1]
        enc_chunk = prompt_chunk + completion_chunk
        prompt_length = len(prompt_chunk)
        completion_length = len(completion_chunk)
        all_length = len(enc_chunk)
    elif prompt_length + completion_length > max_length:
        # Please try to keep the prompt as much as possible,
        # and only truncate it from the left side if it exceeds the maximum length alone.
        if prompt_length > max_length:
            enc_chunk = prompt_chunk + completion_chunk
            enc_chunk = enc_chunk[-max_length:]
            completion_length = len(completion_chunk)
            prompt_length = max_length - completion_length
        # When the completion is too long or the combination of the prompt and completion exceeds the maximum length,
        # truncate from the completion.
        else:
            enc_chunk = prompt_chunk + completion_chunk
            enc_chunk = enc_chunk[:max_length]
            prompt_length = len(prompt_chunk)
            completion_length = max_length - prompt_length
        all_length = len(enc_chunk)
    else:
        # For other cases, keep the original logic as is.
        # padding to the last
        padding_length = max_length - prompt_length - completion_length
        padding_chunk = [tokenizer.eod] * (padding_length)
        enc_chunk = prompt_chunk + completion_chunk + padding_chunk
        all_length = len(enc_chunk) - len(padding_chunk)
    assert len(enc_chunk) == max_length
    assert completion_length + prompt_length <= max_length
    return {'ids': enc_chunk, 'length': all_length}


def build_train_valid_test_datasets_for_rm(input_file, tokenizer, max_length=2048):
    train_ds = RewardDataset(input_file[0], tokenizer, max_length)
    valid_ds = RewardDataset(input_file[1], tokenizer, max_length)
    test_ds = None
    return (train_ds, valid_ds, test_ds)


def padding_sequence(input_list, pad_tok=0):
    if len(input_list[0].shape) > 1:
        max_len = max(len(i) for i in input_list)
        padded_seq = np.ones((len(input_list), max_len) + input_list[0].shape[1:]) * pad_tok
    else:
        max_len = max(len(i) for i in input_list)
        padded_seq = np.ones((len(input_list), max_len)) * pad_tok

    for i, item in enumerate(input_list):
        padded_seq[i, :len(item)] = item

    return padded_seq


def batchify_to_max_length(sample):
    text = torch.stack([torch.LongTensor(sample["data"][idx]['ids']) for idx in range(len(sample["data"]))], 0)
    all_length = np.array([sample["data"][idx]['length'] for idx in range(len(sample["data"]))])  ## pad with 0
    all_score = np.array(sample["scores"])  ## pad with -100

    num_responses = np.array(sample['num_response'])

    output_batch = {
        "text": text,
        "all_length": torch.from_numpy(all_length).long(),
        "num_responses": torch.from_numpy(num_responses),
        "all_score": torch.from_numpy(all_score).long(),
    }

    return output_batch
