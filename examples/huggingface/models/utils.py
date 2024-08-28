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
"""The code is modified from https://github.com/OpenRLHF/OpenRLHF"""

from typing import Optional, Tuple
import os
from pathlib import Path

import torch
from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoConfig, AutoModel
import torch.nn as nn
from transformers.deepspeed import HfDeepSpeedConfig

def blending_datasets(
    datasets,
    probabilities,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
):
    datasets = datasets.split(",")
    if len(datasets) > 1:
        probabilities = list(map(float, probabilities.split(",")))
        assert len(probabilities) == len(datasets)
    else:
        probabilities = [1.0]

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        dataset_subfold_list = dataset.split("@")
        if os.path.isdir(dataset):
            data = load_dataset(dataset)
        # local dir with python script or common local file
        elif os.path.isdir(os.path.join(os.getcwd(), dataset)) or dataset.endswith(
            (".json", ".jsonl", ".csv", ".parquet", ".txt")
        ):
            if dataset.endswith((".json", ".jsonl", ".csv", ".parquet", ".txt")):
                files = dataset
                data_type = os.path.splitext(files)[1][1:]
            else:
                path = Path(dataset)
                script = [str(file.resolve()) for file in Path(path).rglob("*.py")]
                extensions = ("*.json", "*.jsonl", "*.csv", "*.parquet", "*.txt")
                files = [str(file) for ext in extensions for file in Path(path).rglob(ext)]
                print(f"script: {script}")
                print(f"files: {files}")
                # For dir, follow python script or first file type
                data_type = script[0] if len(script) == 1 else os.path.splitext(files[0])[1][1:]
            # reformat data type
            if data_type in ["json", "jsonl"]:
                data_type = "json"
            elif data_type == "txt":
                data_type = "text"
            elif data_type.endswith(".py"):
                # load local dir with python script
                files = None
            if data_type.endswith(".py"):
                print(f"load {dataset} with script {data_type}")
            else:
                print(f"load {files} from {dataset}")
            data = load_dataset(data_type, data_files=files)
        elif len(dataset_subfold_list) == 2:
            dataset = dataset_subfold_list[0]
            subfold = dataset_subfold_list[1]
            data = load_dataset(dataset, data_dir=subfold.strip())
        elif len(dataset_subfold_list) == 1:
            dataset = dataset_subfold_list[0]
            data = load_dataset(dataset)
        else:
            raise Exception(f"Dataset Name {dataset}: Format error")

        if "train" in data:
            train_data_list.append(data["train"].select(range(min(max_count, len(data["train"])))))
        else:
            train_data_list.append(data.select(range(min(max_count, len(data)))))  # train will contains eval? TODO

        if return_eval:
            max_count01 = int(max_count * 0.1)
            if "test" in data:
                eval_data = data["test"].select(range(min(max_count01, len(data["test"]))))
            elif "validation" in data:
                eval_data = data["validation"].select(range(min(max_count01, len(data["validation"]))))
            elif "train" in data:
                eval_data = data["train"].select(range(min(max_count01, int(len(data["train"]) * 0.01))))
            else:
                eval_data = data.select(range(min(int(max_count01), int(len(data) * 0.01))))
            eval_data_list.append(eval_data)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


# Construct transformer with a value head for sequence classification.
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    head_prefix="value_head",
    device_map=None,
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Either "reward" or "critic.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        normalize_reward (bool, optional): Whether normalize reward. Defaults to False.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto
            multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.

    Returns:
        nn.Module: pretrained transformer model.
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    try:
        base_class = AutoModel._model_mapping[type(config)]
        base_pretrained_class = base_class.__base__
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class, head_prefix)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class, head_prefix)
    except Exception as e:
        print("Failed to load from AutoModel, construct from modelling file.")
        module_file, causal_model_name = config.auto_map["AutoModelForCausalLM"].split(".")

        # special case
        if causal_model_name == "QWenLMHeadModel":
            auto_model_name = "QWenModel"
            pretrained_model_name = "QWenPreTrainedModel"
        elif causal_model_name == "InternLMForCausalLM":
            auto_model_name = "InternLMModel"
            pretrained_model_name = "InternLMPreTrainedModel"
        else:
            if "AutoModel" not in config.auto_map:
                auto_model_name = causal_model_name.split("For")[0] + "Model"
            else:
                auto_model_name = config.auto_map["AutoModel"].split(".")[1]
            pretrained_model_name = causal_model_name.split("For")[0] + "PreTrainedModel"

        logger.info(f"BASE_MODEL_CLASS: {auto_model_name}, PRETRAINED_MODEL_CLASS: {pretrained_model_name}")

        base_pretrained_class = get_class_from_dynamic_module(
            f"{module_file}.{pretrained_model_name}", model_name_or_path
        )
        base_class = get_class_from_dynamic_module(f"{module_file}.{auto_model_name}", model_name_or_path)
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class, head_prefix)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class, head_prefix)

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
        **kwargs,
    )

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model(base_pretrained_model, base_llm_model, head_prefix="value_head"):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.head_prefix = head_prefix
            setattr(self, head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.head_prefix)(last_hidden_states).squeeze(-1)

            # left padding in training mode
            if self.training:
                reward = values[:, -1]
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

                # normalize reward in eval mode
                if self.normalize_reward:
                    reward = (reward - self.mean) / self.std
            if return_output:
                return reward, outputs
            else:
                return reward

    return RewardModel


def _get_critic_model(base_pretrained_model, base_llm_model, head_prefix="value_head"):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.head_prefix = head_prefix
            setattr(self, head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.head_prefix)(last_hidden_states).squeeze(-1)[:, :-1]
            num_actions = action_mask.size(1)

            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            if return_output:
                return outputs if num_actions is None else (values[:, -num_actions:], outputs)
            else:
                return values[:, -num_actions:]

    return CriticModel

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)
    else:
        return (tensor * mask).sum() / mask.sum()

class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss

@torch.no_grad()
def get_advantages_and_returns(
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

