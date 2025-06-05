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
from typing import List, Optional, Union, Any
from omegaconf import MISSING

from chatlearn.utils.constant import (RAY_PG_STRATEGY,
    PARAM_SYNC_COMM_TYPE, ROUTED_EXPERT_REGROUPING_COMM_TYPE)


@dataclass
class RuntimeEnvConfig:
    """Runtime env config, you can refer https://docs.ray.io/en/latest/ray-core/handling-dependencies.html for more information."""
    excludes: List[str] = field(
        default_factory=lambda: ["*pt", "logs", "tensorboards", ".nfs*"],
        metadata={"help": "excludes files from packaging"}
    )
    platform: str = field(
        default="DLC",
        metadata={"help": "Platform to run the model. Default is DLC."}
    )
    pip: List[str] = field(
        default_factory=list,
        metadata={"help": "pip install packages"}
    )
    working_dir: Optional[str] = field(
        default=None,
        metadata={"help": "working directory"}
    )
    py_modules: List[str] = field(
        default_factory=list,
        metadata={"help": "python modules"}
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
    num_episode: int = field(
        default=MISSING,
        metadata={"help": "[required] number of episodes. One episode includes a inference and training loop."}
    )
    sample_per_episode: int = field(
        default=MISSING,
        metadata={"help": "[required] number of samples per episode."}
    )
    num_training_epoch: int = field(
        default=1,
        metadata={"help": "[optional] number of training epoch per episode. default set to 1."}
    )
    train_micro_batch_size: int = field(
        default=MISSING,
        metadata={"help": "[required] training micro batch size."}
    )
    train_global_batch_size: int = field(
        default=MISSING,
        metadata={"help": "[required] training global batch size."}
    )
    save_episode_interval: int = field(
        default=MISSING,
        metadata={"help": "save checkpoint per `save_episode_interval` episodes."}
    )
    log_interval: int = field(
        default=1,
        metadata={"help": "[optional] log time and memory per `log_interval` iterations."}
    )
    data_path: str = field(
        default=MISSING,
        metadata={"help": "[required]: data_path for dataset or a List of data_path for different kind of datasets split by ,"}
    )
    data_ratio: List[int] = field(
        default_factory=list, # origin None
        metadata={"help": "the ratio for each kind of data_path in a training episode"}
    )
    data_shuffle: bool = field(
        default=True,
        metadata={"help": "shuffle in each epoch of dataset, default: True"}
    )
    data_rerank: bool = field(
        default=True,
        metadata={"help": "rerank batch of data by row"}
    )
    colocation: List[str] = field(
        default_factory=list,
        metadata={"help": "colocate models into the same device"}
    )
    eval_episode_interval: int = field(
        default=0,
        metadata={"help": "[optional]: eval every N episode, if 0, will not eval"}
    )
    enable_resume_training: bool = field(
        default=True,
        metadata={"help": "[optional]: enable resume training when data checkpoint is set"}
    )
    data_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "[optional]: checkpoint for dataloader"}
    )
    max_data_ckpt_nums: Optional[int] = field(
        default=None,
        metadata={"help": "[optional]: max data checkpoint nums"}
    )
    load_data_checkpoint_iteration: Optional[int] = field(
        default=None,
        metadata={"help": "[optional]: load data checkpoint from iteration"}
    )
    stream_data_loader_type: str = field(
        default="fixed",
        metadata={"help": "stream_data_loader type, fixed or dynamic"}
    )
    debug: bool = field(
        default=False,
        metadata={"help": "whether log debug infromation or not"}
    )
    nsys: bool = field(
        default=False,
        metadata={"help": "whether enable nsys nvtx"}
    )
    profiler_dir: Optional[str] = field(
        default=None,
        metadata={"help": "profiler dir"}
    )
    coalesced_buffer_mb: int = field(
        default=100,
        metadata={"help": "coalesce_buffer size in mb"}
    )
    concurrent_comm: bool = field(
        default=True,
        metadata={"help": "whether concurrent parameter sync or not, for megatron to vllm"}
    )
    param_sync_comm_type: str = field(
        default=PARAM_SYNC_COMM_TYPE.BROADCAST.value,
        metadata={"help": "parameter sync communication type, broadcast/p2p"}
    )
    param_sync_max_workers: Optional[int] = field(
        default=None,
        metadata={"help": "parameter sync max workers"}
    )
    routed_expert_regrouping_comm_type: str = field(
        default=ROUTED_EXPERT_REGROUPING_COMM_TYPE.ALLTOALL,
        metadata={"help": "communication type to regroup routed experts, allgather/alltoall"}
    )
    max_replay_episode: int = field(
        defautl=0,
        metadata={"help": "max number of replay episodes, if `max_replay_episode` is set to -1, then replay all episodes \
                        if `max_replay_episode` is set to 0, then replay is disabled"}
    )
    replay_episode_offset: int = field(
        default=0,
        metadata={"help": "replay after n episodes"}
    )
    eval_data_path: str = field(
        default=MISSING,
        metadata={"help": "Path to the evaluation data file. Required."}
    )
