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
"""common configs"""

from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class RuntimeEnvConfig:
    """RuntimeEnvConfig"""

    platform: str = field(
        default="DLC",
        metadata={"help": "Platform to run the model. Default is DLC."}
    )


@dataclass
class BaseModelConfig:
    """BaseModelConfig"""

    seed: int = field(
        default=1234,
        metadata={"help": "Random seed. Default is 1234."}
    )


@dataclass
class PolicyConfig(BaseModelConfig):
    """PolicyConfig"""

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
    """RewardConfig"""

    num_cpus: int = field(
        default=2,
        metadata={"help": "Number of CPUs to use. Default is 1."}
    )


@dataclass
class RefPolicyConfig(BaseModelConfig):
    """RefPolicyConfig"""

    fsdp_size: int = field(
        default=-1,
        metadata={"help": "FSDP size. Default is -1."}
    )


@dataclass
class PolicyTrainerConfig(BaseModelConfig):
    """PolicyTrainerConfig"""

    free_memory: bool = field(
        default=True,
        metadata={"help": "Whether to free memory. Default is True."}
    )


@dataclass
class RuntimeConfig:
    """RuntimeConfig"""

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
