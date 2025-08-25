"""Configs for policy model"""
from dataclasses import dataclass, field

from .base import BaseConfig, BaseModelConfig

@dataclass
class RolloutManagerConfig(BaseModelConfig):
    """The config for rollout models. Currently this config is
    shared between vLLM and SGLang.
    """
    max_rollout_round: int = field(
        default=2, metadata={"help": "Max rollout round for one sample"}
    )
    max_gen_len: int = field(
        default=2048, metadata={"help": "max response token length for one sample"}
    )
    num_inference_per_prompt: int = field(
        default=32, metadata={"help": "number of response for per prompt"}
    )
    mini_response_per_prompt: int = field(
        default=16, metadata={"help": "when number of finished rollout for prompt is larger then this threshold. \
            move these responses for training"}
    )
    enable_thinking: bool = field(
        default=False, metadata={"help": "whether enable think or not"}
    )
