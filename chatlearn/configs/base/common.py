"""common configs"""
from dataclasses import dataclass, field

from .base_config import BaseConfig
from .base_model_config import BaseModelConfig

@dataclass
class OptimizerConfig(BaseConfig):
    """OptimizerConfig"""
    clip_grad: float = field(
        default=1.0, metadata={"help": "Gradient clipping based on global L2 norm."}
    )
    lr: float = field(default=2e-6, metadata={"help": "Initial learning rate."})
    min_lr: float = field(
        default=0, metadata={"help": "Minimum value for learning rate."}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay coefficient for L2 regularization."},
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={
            "help": "First coefficient for computing running averages of gradient and its square"
        },
    )
    adam_beta2: float = field(
        default=0.95,
        metadata={
            "help": "Second coefficient for computing running averages of gradient and its square"
        },
    )

@dataclass
class RefPolicyConfig(BaseModelConfig):
    """Common configs for reference policy model"""

@dataclass
class RewardConfig(BaseModelConfig):
    """Common configs for reward model"""

@dataclass
class PolicyTrainerConfig(BaseModelConfig):
    """PolicyTrainerConfig"""
    optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig, metadata={"help": "optimizer config"}
    )
    # NOTE: shared config between all GRPO Trainer Class. Move elsewhere if needed
    entropy_coef: float = field(
        default=0.0, metadata={"help": "entropy regularization"}
    )
    kl_coef: float = field(
        default=0.0, metadata={"help": "kl regularization"}
    )
    pos_clip_ratio: float = field(default=0.2)
    neg_clip_ratio: float = field(default=0.2)
    diff_clip_ratio: float = field(default=10)
    final_clip_ratio: float = field(default=3)
    use_group_sequence_policy: bool = field(
        default=False, metadata={"help": "whether to use group sequence policy optimization"}
    )
    train_micro_batch_size: int = field(
        default=1, metadata={"help": "train microbatch size for gradient accumulation"}
    )
