"""prompt dataset"""

import copy
from collections import defaultdict
from typing import List, Dict

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer

from chatlearn.utils.utils import multi_thread_data_processing


class VLLMPromptPipeline(Dataset):
    """
    process this format
    {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'answer': answer_raw,
            "question": question_raw,
        }
    }
    self.data format
    {"input_ids": prompt_ids, "prompt": prompt}
    """

    def __init__(self, data_list: List[Dict], seq_length: int, tokenizer: AutoTokenizer = None, num_inference_per_prompt: int = 1):# pylint: disable=super-init-not-called
        super().__init__()

        self.tokenizer = tokenizer
        self.data = []

        for data_item in data_list:
            prompt = data_item["prompt"]
            data_source = data_item.get("data_source", "")
            ground_truth = data_item['reward_model']['ground_truth']
            if isinstance(prompt, list):
                prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            input_ids = self.tokenizer.encode(prompt)
            processed_data = {"input_ids": input_ids, "prompt": prompt, "data_source": data_source, "ground_truth": ground_truth}
            if seq_length > len(input_ids):
                self.data.extend([processed_data]*num_inference_per_prompt)
        
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