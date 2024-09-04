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
"""prompt dataset"""

import copy
from collections import defaultdict
from typing import List

import torch
from megatron.training import get_args
from torch.utils.data import Dataset
import torch.nn.functional as F
from examples.megatron.models.utils import get_eos_id

from chatlearn.utils.utils import multi_thread_tokenize

def zero_pad_sequences(sequences, side: str = "right", value=0, pad_to_seq_length=False):
    assert side in ("left", "right")
    if pad_to_seq_length: # pad to args.seq_length
        args = get_args()
        max_len = args.seq_length
    else: # pad to the max sequence length of the current batch
        max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


class PromptPipeline(Dataset):
    """
    a dataset of list of no padded prompt tensors
    truncted to max_prompt_length from right
    """

    def __init__(self, prompts: List[str], max_prompt_length: int, tokenizer=None):
        super().__init__()

        for p in prompts:
            assert len(p) > 0, "Got empty prompt"
        assert max_prompt_length > 0, \
            "Prompt length for RLHF/OnlineDPO/GRPO trainer must be an integer greater than 0"

        if len(prompts[0]) == 3:
            prompt_encodings = [tokenizer.tokenize(prompt)[:max_prompt_length] for prompt, _, _ in prompts]
        else:
            prompt_encodings = [tokenizer.tokenize(prompt)[:max_prompt_length] for prompt in prompts]
        prompt_id_tensors = [torch.tensor(p_encoding, dtype=torch.long) for p_encoding in prompt_encodings]

        # dup dataset if num_inference_per_prompt
        self.data = []
        prompts = [{"input_ids": prompt_tensor} for prompt_tensor in prompt_id_tensors]
        for p in prompts:
            dup = [copy.deepcopy(p) for i in range(get_args().num_inference_per_prompt)]
            self.data.extend(dup)

        self.tokenizer = tokenizer

    def __getitem__(self, ix: int):
        return self.data[ix]

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn(self, samples):
        collate_dict = defaultdict(list)

        # Loop over the samples and append each tensor value to the corresponding list
        for sample in samples:
            for key in sample.keys():
                collate_dict[key].append(sample[key])

        # Return the collate_dict
        return collate_dict


class DPOPromptPipeline(PromptPipeline):
    """
    a dataset of list of no padded prompt tensors
    truncted to max_prompt_length from right
    """

    def __init__(self, prompts: List[str], max_seq_length: int, tokenizer=None):# pylint: disable=super-init-not-called

        self.data = []
        for prompt, chosen, rejected in prompts:
            chosen = prompt + chosen
            rejected = prompt + rejected
            chosen_token = tokenizer.tokenize(chosen)[:max_seq_length]
            reject_token = tokenizer.tokenize(rejected)[:max_seq_length]
            chosen_token[-1] = get_eos_id(tokenizer)
            reject_token[-1] = get_eos_id(tokenizer)
            prompt_id_len = len(tokenizer.tokenize(prompt))
            # has at least one token from positive/negative responses
            if prompt_id_len >= max_seq_length:
                continue
            chosen_token = torch.tensor(chosen_token, dtype=torch.long)
            chosen_mask = torch.ones((1, chosen_token.shape[-1]))
            reject_token = torch.tensor(reject_token, dtype=torch.long)
            reject_mask = torch.ones((1, reject_token.shape[-1]))
            sample = (chosen_token, chosen_mask, reject_token, reject_mask, prompt_id_len)
            self.data.append(sample)
        self.tokenizer = tokenizer

    def collate_fn(self, samples):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        prompt_id_lens = []
        for chosen_id, chosen_mask, reject_id, reject_mask, prompt_id_len in samples:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(reject_mask)
            prompt_id_lens.append(prompt_id_len)

        chosen_ids = zero_pad_sequences(chosen_ids, value=get_eos_id(self.tokenizer), pad_to_seq_length=True)
        chosen_masks = zero_pad_sequences(chosen_masks, pad_to_seq_length=True)
        reject_ids = zero_pad_sequences(reject_ids, value=get_eos_id(self.tokenizer), pad_to_seq_length=True)
        rejects_masks = zero_pad_sequences(rejects_masks, pad_to_seq_length=True)
        return {
            "chosen": chosen_ids,
            "chosen_mask": chosen_masks,
            "rejected": reject_ids,
            "rejected_mask": rejects_masks,
            "prompt_id_lens": torch.tensor(prompt_id_lens, dtype=torch.long)
        }


class VLLMPromptPipeline(PromptPipeline):
    """
    a dataset of list of no padded prompt tensors
    truncted to max_prompt_length from right
    """

    def __init__(self, prompts: List, max_prompt_length: int, tokenizer=None, prompt_key=None):# pylint: disable=super-init-not-called
        for p in prompts:
            assert len(p) > 0, "Got empty prompt"
        assert max_prompt_length > 0, \
            "Prompt length for RLHF/OnlineDPO trainer must be an integer greater than 0"
        if prompt_key is None:
            if len(prompts[0]) == 3:
                valid_prompts = [p for p, _, _ in prompts]
                self.prompts = multi_thread_tokenize(valid_prompts, tokenizer, max_prompt_length)
            else:
                self.prompts = [(prompt, tokenizer.encode(prompt)[:max_prompt_length]) for prompt in prompts]
            self.data = []
            for prompt, prompt_ids in self.prompts:
                p = {"input_ids": prompt_ids, "prompt": prompt}
                self.data.extend([copy.deepcopy(p)])
        else:
            for prompt in prompts:
                prompt["input_ids"] = tokenizer.encode(prompt[prompt_key])[:max_prompt_length]
                if 'prompt' != prompt_key:
                    prompt['prompt'] = prompt[prompt_key]
            self.data = prompts
        self.tokenizer = tokenizer
