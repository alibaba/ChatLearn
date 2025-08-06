"""prompt dataset"""

from typing import List, Dict

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PromptPipeline(Dataset):
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

    def __init__(
        self,
        data_list: List[Dict],
        seq_length: int,
        tokenizer: AutoTokenizer = None,
        enable_thinking=False
    ):  # pylint: disable=super-init-not-called
        super().__init__()

        self.tokenizer = tokenizer
        self.data = []

        for data_item in data_list:
            prompt = data_item["prompt"]
            data_source = data_item.get("data_source", "")
            ground_truth = data_item["reward_model"]["ground_truth"]
            if isinstance(prompt, list):
                prompt = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            input_ids = self.tokenizer.encode(prompt)
            processed_data = {
                "input_ids": input_ids,
                "prompt": prompt,
                "data_source": data_source,
                "ground_truth": ground_truth,
            }
            if seq_length > len(input_ids):
                self.data.append(processed_data)

    def __getitem__(self, ix: int):
        return self.data[ix]

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn(self, samples):
        return samples
