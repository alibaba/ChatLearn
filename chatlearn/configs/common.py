from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class RuntimeEnvConfig:
    platform: str = field(
        default="DLC",
        metadata={"help": "Platform to run the model. Default is DLC."}
    )


@dataclass
class BaseModelConfig:
    seed: int = field(
        default=1234,
        metadata={"help": "Random seed. Default is 1234."}
    )


@dataclass
class PolicyConfig(BaseModelConfig):
    num_gpus: int = field(
        default=1,
        metadata={"help": "Number of GPUs to use. Default is 1."}
    )
    trainable: bool = field(
        default=False,
        metadata={"help": "Whether the policy is trainable. Default is False."}
    )


@dataclass
class RewardConfig(BaseModelConfig):
    num_cpus: int = field(
        default=2,
        metadata={"help": "Number of CPUs to use. Default is 1."}
    )


@dataclass
class RefPolicyConfig(BaseModelConfig):
    fsdp_size: int = field(
        default=-1,
        metadata={"help": "FSDP size. Default is -1."}
    )


@dataclass
class PolicyTrainerConfig(BaseModelConfig):
    free_memory: bool = field(
        default=True,
        metadata={"help": "Whether to free memory. Default is True."}
    )


@dataclass
class RuntimeConfig:
    colocation: list[str] = field(
        default_factory=list,
        metadata={"help": "List of modules to colocate. Default is empty."}
    )
    data_path: str = field(
        default=MISSING,
        metadata={"help": "Path to the data file. Required."}
    )
    eval_data_path: str = field(
        default=MISSING,
        metadata={"help": "Path to the evaluation data file. Required."}
    )

