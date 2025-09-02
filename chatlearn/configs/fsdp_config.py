"""Config classes for FSDP"""
from dataclasses import dataclass, field

from .base import BaseConfig, PolicyTrainerConfig, RefPolicyConfig, BaseModelConfig


__all__ = ['FSDPPolicyTrainerConfig', 'FSDPRefPolicyConfig']


@dataclass
class FSDPConfig(BaseConfig):
    """FSDP-related configurations"""
    fsdp_size: int = field(default=1, metadata={"help": "FSDP parallel size"})
    ulysses_sequence_parallel_size: int = field(
        default=1,
        metadata={"help": "ulysses sequence parallel size used for fsdp train backend"},
    )
    meta_init: bool = field(
        default=False, metadata={"help": "Whether to use meta init for FSDP. When using groupgemm, recommend enable meta init"}
    )
    groupgemm: bool = field(
        default=False, metadata={"help": "Whether to use groupgemm patch for moe, now only support qwen3moe model"}
    )

    use_expandable_segments: bool = field(
        default=False, metadata={"help": "Whether to use expandable_segments in PYTORCH_CUDA_ALLOC_CONF, \
            avoid big reseverd memory in ref and policy trainer worker, expandable_segments should be False \
            while in parameter sync for efficiency"}
    )

    def _validate_impl(self):
        assert self.num_gpu > 0, "FSDP requires at least one GPU"
        assert self.num_gpu % self.ulysses_sequence_parallel_size == 0, \
            "models.policy_trainer.num_gpu must be divisible by models.policy_trainer.ulysses_sequence_parallel_size"

    def _post_init_impl(self):
        if isinstance(self, BaseModelConfig):
            # NOTE: currently fsdp_size hard-coded
            self.fsdp_size, self.num_replica = self.num_gpu, 1
            self.replica_dp_size = self.num_gpu // (self.ulysses_sequence_parallel_size * self.num_replica)

@dataclass
class FSDPRefPolicyConfig(RefPolicyConfig, FSDPConfig):
    """Config for FSDP reference policy model"""

@dataclass
class FSDPPolicyTrainerConfig(PolicyTrainerConfig, FSDPConfig):
    """Config for FSDP policy trainer"""
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "whether gradient checkpointing"}
    )
    save_hf: bool = field(default=True, metadata={"help": "whether to save transformer-style checkpoint"})
