"""Configs for policy model"""
from dataclasses import dataclass, field

from .base import BaseConfig, BaseModelConfig

@dataclass
class RolloutConfig(BaseConfig):
    """The config for rollout models. Currently this config is
    shared between vLLM and SGLang.
    """
    is_sync_mode: bool = field(
        default=True, metadata={"help": "whether use sync rollout or async rollout. warning: vLLM only support sync mode"}
    )
    tensor_model_parallel_size: int = field(
        default=1, metadata={"help": "tensor model parallel size"}
    )
    pipeline_model_parallel_size: int = field(
        default=1, metadata={"help": "pipeline model parallel size"}
    )
    expert_model_parallel_size: int = field(
        default=1, metadata={"help": "expert model parallel size for Rollout Engine"}
    )
    max_new_tokens: int = field(
        default=2048,
        metadata={"help": "length of response"}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for sample train data"}
    )
    top_p: float = field(default=1.0, metadata={"help": "top_p for sample train data"})
    top_k: int = field(default=-1, metadata={"help": "top_k for sample train data"})
    presence_penalty: float = field(
        default=0.0, metadata={"help": "presence_penalty for sample train data"}
    )
    frequency_penalty: float = field(
        default=0.0, metadata={"help": "frequency_penalty for sample train data"}
    )
    repetition_penalty: float = field(
        default=1.0, metadata={"help": "repetition_penalty for sample train data"}
    )
    eval_temperature: float = field(
        default=0.6, metadata={"help": "temperature for sample eval data"}
    )
    eval_top_p: float = field(
        default=0.95, metadata={"help": "top_p for sample eval data"}
    )
    eval_top_k: int = field(default=20, metadata={"help": "top_k for sample eval data"})
    eval_presence_penalty: float = field(
        default=0.0, metadata={"help": "presence_penalty for sample eval data"}
    )
    eval_frequency_penalty: float = field(
        default=0.0, metadata={"help": "frequency_penalty for sample eval data"}
    )
    eval_repetition_penalty: float = field(
        default=1.0, metadata={"help": "repetition_penalty for sample eval data"}
    )
    # TODO: remove these two keys
    vllm_prompt_key: str = field(default="prompt", metadata={"help": "vllm_prompt_key"})
    vllm_input_ids_key: str = field(
        default="input_ids", metadata={"help": "vllm_input_ids_key"}
    )
    # TODO: remove these two keys
    max_num_batched_tokens: int = field(
        default=32768, metadata={"help": "max_num_batched_tokens"}
    )
    max_seq_len_to_capture: int = field(
        default=2348, metadata={"help": "max_seq_len_to_capture"}
    )
    gpu_memory_utilization: float = field(
        default=0.8, metadata={"help": "gpu_memory_utilization"}
    )
    enforce_eager: bool = field(default=False, metadata={"help": "enforce_eager"})

@dataclass
class PolicyConfig(BaseModelConfig, RolloutConfig):
    """General Config Class for Policy/Rollout Model"""
    num_inference_per_prompt: int = field(
        default=32, metadata={"help": "number of response for per prompt"}
    )
    seq_length: int = field(
        default=2048, metadata={"help": "length of prompt + response"}
    )
    enable_thinking: bool = field(
        default=False, metadata={"help": "whether enable think or not"}
    )
    enable_stage_resume: bool = field(
        default=False, metadata={"help": "enable_stage_resume"}
    )

    def _validate_impl(self):
        assert self.num_gpu % self.tensor_model_parallel_size == 0, \
            "models.policy.num_gpu must be divisible by tensor_model_parallel_size"
        assert self.num_gpu > 0, "Policy model requires at least one GPU"
        assert not self.trainable, "Policy model does not support training"
        assert self.expert_model_parallel_size == 1, "Expert Parallel of Policy model is not supported"
        assert self.pipeline_model_parallel_size == 1, "Pipeline Parallel of Policy model is not supported"
        assert self.num_gpu % self.num_replica == 0, \
            "The GPUs assigned to megatron model must be divisible by num_replica"   

    def _post_init_impl(self):
        self.num_replica = self.num_gpu // (
            self.tensor_model_parallel_size *
            self.expert_model_parallel_size *
            self.pipeline_model_parallel_size
        )
        self.replica_dp_size = 1
