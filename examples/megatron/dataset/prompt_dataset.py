import copy
from collections import defaultdict
from typing import List

import torch
from megatron import get_args
from torch.utils.data import DataLoader, Dataset


def custom_collate(samples):
    # Initialize an empty dict for lists
    dict = defaultdict(list)

    # Loop over the samples and append each tensor value to the corresponding list
    for sample in samples:
        for key in sample.keys():
            dict[key].append(sample[key])

    # Return the dict    
    return dict


class PromptPipeline(Dataset):
    """
    a dataset of list of no padded prompt tensors
    truncted to max_prompt_length from right
    """

    def __init__(self, prompts: List[str], max_prompt_length: int, tokenizer=None):
        super().__init__()

        for p in prompts:
            assert len(p) > 0, "Got empty prompt"

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

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        collate_fn = custom_collate
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )
