import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

def exist_and_not_none(d, key):
    return key in d and d[key] is not None

def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    apply_chat_template=None,
    is_dpo=False,
) -> str:
    if apply_chat_template:
        if prompt_key:
            prompt = apply_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
            chosen = apply_chat_template(data[prompt_key] + data[chosen_key], tokenize=False)[len(prompt) :]
            rejected = apply_chat_template(data[prompt_key] + data[rejected_key], tokenize=False)[len(prompt) :]
        else:
            prompt = ""
            chosen = apply_chat_template(data[chosen_key], tokenize=False)
            rejected = apply_chat_template(data[rejected_key], tokenize=False)

            if is_dpo:
                prompt = apply_chat_template(data[chosen_key][:-1], tokenize=False, add_generation_prompt=True)
                chosen = chosen[len(prompt) :]
                rejected = rejected[len(prompt) :]
    else:
        if prompt_key:
            prompt = data[prompt_key]
            start_str = input_template.split("{}")[0]
            end_str = input_template.split("{}")[1]
            if input_template:
                if not (prompt.startswith(start_str) and prompt.endswith(end_str)):
                    prompt = input_template.format(prompt)
        else:
            prompt = ""
        chosen = data[chosen_key]
        rejected = data[rejected_key]

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    return prompt, chosen, rejected, margin

class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        args,
        tokenizer,
        max_length: int,
        input_template=None,
        is_dpo=False,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.args = args

        self.prompts = []
        self.chosens = []
        self.rejects = []
        if self.is_dpo:
            self.prompt_ids_lens = []
        else:
            self.margins = []

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_dpo = is_dpo

        prompt_key = getattr(self.args, "prompt_key", None)
        chosen_key = getattr(self.args, "chosen_key", None)
        rejected_key = getattr(self.args, "rejected_key", None)
        apply_chat_template = getattr(self.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        for data in tqdm(dataset, desc="Tokenizing"):
            prompt, chosen, reject, margin = preprocess_data(
                data, input_template, prompt_key, chosen_key, rejected_key, apply_chat_template, self.is_dpo
            )
            if self.is_dpo:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                else:
                    self.prompt_ids_lens.append(prompt_ids_len)
            else:
                self.margins.append(margin)

            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject = self.prompts[idx], self.chosens[idx], self.rejects[idx]
        if self.is_dpo:
            extra = self.prompt_ids_lens[idx]
        else:
            extra = self.margins[idx]

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return {
            "chosen_input_ids": chosen_token["input_ids"],
            "chosen_attention_mask": chosen_token["attention_mask"],
            "reject_input_ids": reject_token["input_ids"],
            "reject_attention_mask": reject_token["attention_mask"],
            "extra": extra,
        }

    def collate_fn(self, item_dict):
        batches = defaultdict(list)
        for data in item_dict:
            for key, value in data.items():
                batches[key].append(value)

        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        for key, data in batches.items():
            if key.endswith("input_ids"):
                batches[key] = zero_pad_sequences(data, side=padding_side, value=self.tokenizer.pad_token_id)
            elif key.endswith("mask"):
                batches[key] = zero_pad_sequences(data, side=padding_side)
        return batches
