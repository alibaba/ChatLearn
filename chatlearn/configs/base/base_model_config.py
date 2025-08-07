"""Basic Configs for a ChatLearn Module"""
from dataclasses import dataclass, field
from typing import Optional

from .base_config import BaseConfig

@dataclass
class FreeGpuMemoryConfig(BaseConfig):
    """Configs to on"""
    offload_weights: bool = field(
        default=False,
        metadata={
            "help": "whether offload weights to cpu, used for inference and trainer"
        },
    )
    free_grad_buffers: bool = field(
        default=False,
        metadata={"help": "whether free grad buffers, only used for Mcore"},
    )
    offload_optimizer_states: bool = field(
        default=False,
        metadata={"help": "whether offload optimizer states to cpu, used for trainer"},
    )

@dataclass
class BaseModelConfig(BaseConfig):
    """BaseModelConfig"""

    num_gpu: int = field(
        default=0, metadata={"help": "number of GPU used for one model, default 0"}
    )
    num_cpu: int = field(
        default=0, metadata={"help": "number of CPU used for one model, default 0"}
    )
    gpu_per_process: Optional[int] = field(
        default=None,
        metadata={
            "help": "gpu per process, e.g., for PyTorch DDP, Megatron, `gpu_per_process` is set to 1"
        },
    )
    cpu_per_process: Optional[int] = field(
        default=None, metadata={"help": "cpu per process"}
    )
    trainable: bool = field(
        default=False, metadata={"help": "whether model is trainable"}
    )

    packing: bool = field(default=False, metadata={"help": "Whether to use sequence packing"})
    max_token_in_packing: int = field(
        default=32768, metadata={"help": "max token in packing when packing is enabled"}
    )

    generation_batch_size: int = field(
        default=1, metadata={"help": "rollout generation batch size"}
    )
    sync_frequency: int = field(
        default=1, metadata={"help": "parameter sync frequency"}
    )
    free_gpu_memory: FreeGpuMemoryConfig = field(
        default_factory=FreeGpuMemoryConfig, metadata={"help": "free gpu memory config"}
    )

    load: Optional[str] = field(default=None, metadata={"help": "path to model weight."})
    seed: int = field(default=1234, metadata={"help": "random seed"})

    replica_dp_size: Optional[int] = field(
        default=None,
        metadata={"help": "The data parallel size in the replica. Maybe None if not needed."}
    )
    num_replica: int = field(
        default=1,
        metadata={"help": "The number of replica of this model."}
    )
    