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
"""prompt dataset"""

import copy
from collections import defaultdict
from typing import List

import torch
from megatron import get_args
from torch.utils.data import Dataset


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
            "Prompt length for PPO trainer must be an integer greater than 0"

        prompt_encodings = [tokenizer.tokenize(prompt)[:max_prompt_length] for prompt in prompts]
        prompt_id_tensors = [torch.tensor(p_encoding, dtype=torch.long) for p_encoding in prompt_encodings]

        # dup dataset if num_inference_per_prompt
        self.prompts_ids = []
        prompts = [{"input_ids": prompt_tensor} for prompt_tensor in prompt_id_tensors]
        for p in prompts:
            dup = [copy.deepcopy(p) for i in range(get_args().num_inference_per_prompt)]
            self.prompts_ids.extend(dup)

        self.tokenizer = tokenizer

    def __getitem__(self, ix: int):
        return self.prompts_ids[ix]

    def __len__(self) -> int:
        return len(self.prompts_ids)

    def collate_fn(self, samples):
        collate_dict = defaultdict(list)

        # Loop over the samples and append each tensor value to the corresponding list
        for sample in samples:
            for key in sample.keys():
                collate_dict[key].append(sample[key])

        # Return the collate_dict
        return collate_dict
