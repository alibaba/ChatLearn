"""Configs for policy model"""
from dataclasses import dataclass, field
from typing import List

from .base import BaseModelConfig

@dataclass
class RolloutManagerConfig(BaseModelConfig):
    """The config for rollout models. Currently this config is
    shared between vLLM and SGLang.
    """
    max_rollout_round: int = field(
        default=2, metadata={"help": "Max rollout round for one sample"}
    )
    max_response_tokens_length: int = field(
        default=2048, metadata={"help": "max length of response"}
    )
    max_prompt_tokens_length: int = field(
        default=2048, metadata={"help": "max length of prompt"}
    )
    num_inference_per_prompt: int = field(
        default=32, metadata={"help": "number of response for per prompt"}
    )
    mini_response_per_prompt: int = field(
        default=16, metadata={"help": "when number of finished rollout for prompt is larger then this threshold. \
            move these responses for training"}
    )
    rollout_ratio: List[float] = field(
        default_factory=lambda: [0.5,0.5], metadata={"help":"rollout ratio for each round, \
            max rollout token for each round_i is max_gen_len * rollout_ratio[i]"}
    )
    enable_thinking: bool = field(
        default=False, metadata={"help": "whether enable think or not"}
    )

    def _validate_impl(self):
        assert len(self.rollout_ratio) == self.max_rollout_round, \
            "Rollout_ratio for each round must be set"
        assert sum(self.rollout_ratio) == 1.0, \
            "Sum of rollout ratio for each round must equal to 1.0"
        assert self.mini_response_per_prompt < self.num_inference_per_prompt, \
            "mini_response_per_prompt must be less than num_inference_per_prompt"
