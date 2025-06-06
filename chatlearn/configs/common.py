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

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Any, Iterator
from omegaconf import MISSING

from chatlearn.utils.constant import (RAY_PG_STRATEGY,
    PARAM_SYNC_COMM_TYPE, ROUTED_EXPERT_REGROUPING_COMM_TYPE)


class BaseConfig:
    # TODO: Unifying Parameter Access Using dataclass approach

    def __getitem__(self, key: str):
        """support args[key]"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """support key in args"""
        return hasattr(self, key)
    
    def get(self, key: str, default=None):
        """support args.get(key)"""
        return getattr(self, key, default)

    def items(self) -> Iterator[tuple[str, Any]]:
        """support args.items()"""
        return asdict(self).items()

    def keys(self) -> Iterator[str]:
        """support args.keys()"""
        return asdict(self).keys()
    
    def values(self) -> Iterator[Any]:
        """support args.values()"""
        return asdict(self).values()


@dataclass
class RuntimeEnvConfig(BaseConfig):
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
class LogConfig(BaseConfig):
    log_dir: str = field(
        default=MISSING,
        metadata={"help": "log dir"}
    )
    # config for tensorboard
    enable_tensorboard: bool = field(
        default=False,
        metadata={"help": "whether enable tensorboard or not"}
    )
    tensorboard_dir: Optional[str] = field(
        default=None,
        metadata={"help": "tensorboard file save dir"}
    )
    # config for wandb
    enable_wandb: bool = field(
        default=False,
        metadata={"help": "whether enable wandb or not"}
    )
    wandb_dir: Optional[str] = field(
        default=None,
        metadata={"help": "wandb file save dir"}
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "wandb project"}
    )
    wandb_id: Optional[str] = field(
        default=None,
        metadata={"help": "wandb id"}
    )
    wandb_name: Optional[str] = field(
        default=None,
        metadata={"help": "wandb name"}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "wandb resume"}
    )

@dataclass
class BaseModelConfig(BaseConfig):
    """BaseModelConfig"""

    num_gpu: int = field(
        default=0,
        metadata={"help": "number of GPU used for one model, default 0"}
    )
    num_cpu: int = field(
        default=0,
        metadata={"help": "number of CPU used for one model, default 0"}
    )
    gpu_per_process: Optional[int] = field(
        default=None,
        metadata={"help": "gpu per process, e.g., for PyTorch DDP, Megatron, `gpu_per_process` is set to 1"}
    )
    cpu_per_process: Optional[int] = field(
        default=None,
        metadata={"help": "cpu per process"}
    )
    trainable: bool = field(
        default=False,
        metadata={"help": "whether model is trainable"}
    )
    tensor_model_parallel_size: int = field(
        default=1,
        metadata={"help": "tensor model parallel size"}
    )
    pipeline_model_parallel_size: int = field(
        default=1,
        metadata={"help": "pipeline model parallel size"}
    )
    expert_model_parallel_size: int = field(
        default=1,
        metadata={"help": "expert model parallel size for Megatron-Core"}
    )
    zero_size: int = field(
        default=1,
        metadata={"help": "zero size"}
    )
    fsdp_size: int = field(
        default=1,
        metadata={"help": "FSDP parallel size"}
    )
    sp_size: int = field(
        default=1,
        metadata={"help": "Sequence parallel size"}
    )
    generation_batch_size: int = field(
        default=1,
        metadata={"help": "rollout generation batch size"}
    )
    offload_optimizer_states: bool = field(
        default=False,
        metadata={"help": "whether offload optimizer states"}
    )
    sync_frequency: int = field(
        default=1,
        metadata={"help": "parameter sync frequency"}
    )
    offload_weights: bool = field(
        default=False,
        metadata={"help": "whether offload weights"}
    )
    free_grad_buffers: bool = field(
        default=False,
        metadata={"help": "whether free grad buffers"}
    )
    free_memory: bool = field(
        default=False,
        metadata={"default": "overall switch for offload optimizer states/weights and free grad buffers"}
    )
    
@dataclass
class PolicyArgsDictConfig(BaseConfig):
    "temp support"
    num_inference_per_prompt: int = field(
        default=32,
        metadata={"help": "number of response for per prompt"}
    )
    seq_length: int = field(
        default=2048,
        metadata={"help": "length of prompt + response"}
    )
    max_new_tokens: int = field(
        default=2048,
        metadata={"help": "length of response"}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for sample train data"}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "top_p for sample train data"}
    )
    top_k: int = field(
        default=-1,
        metadata={"help": "top_k for sample train data"}
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "presence_penalty for sample train data"}
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "frequency_penalty for sample train data"}
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "repetition_penalty for sample train data"}
    )

    eval_temperature: float = field(
        default=0.6,
        metadata={"help": "temperature for sample eval data"}
    )
    eval_top_p: float = field(
        default=0.95,
        metadata={"help": "top_p for sample eval data"}
    )
    eval_top_k: int = field(
        default=20,
        metadata={"help": "top_k for sample eval data"}
    )
    eval_presence_penalty: float = field(
        default=0.0,
        metadata={"help": "presence_penalty for sample eval data"}
    )
    eval_frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "frequency_penalty for sample eval data"}
    )
    eval_repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "repetition_penalty for sample eval data"}
    )
    vllm_prompt_key: str = field(
        default="prompt",
        metadata={"help": "vllm_prompt_key"}
    )
    vllm_input_ids_key: str = field(
        default="input_ids",
        metadata={"help": "vllm_input_ids_key"}
    )
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "whether enable think or not"}
    )
    max_num_batched_tokens: int = field(
        default=32768,
        metadata={"help": "max_num_batched_tokens"}
    )
    max_seq_len_to_capture: int = field(
        default=2348,
        metadata={"help": "max_seq_len_to_capture"}
    )
    enable_stage_resume: bool = field(
        default=False,
        metadata={"help": "enable_stage_resume"}
    )
    gpu_memory_utilization: float = field(
        default=0.8,
        metadata={"help": "gpu_memory_utilization"}
    )
    enforce_eager: bool = field(
        default=False,
        metadata={"help": "enforce_eager"}
    )
    tokenizer: str = field(
        default=MISSING,
        metadata={"help": "tokenizer config path"}
    )
    seed: int = field(
        default=1234,
        metadata={"help": "seed"}
    )
    
    # for debug
    tensor_model_parallel_size: int = field(
        default=1,
        metadata={"help": "tensor model parallel size"}
    )
    pipeline_model_parallel_size: int = field(
        default=1,
        metadata={"help": "pipeline model parallel size"}
    )

@dataclass
class PolicyConfig(BaseModelConfig):
    """PolicyConfig"""

    # args_dict: PolicyArgsDictConfig = field(
    #     default_factory=PolicyArgsDictConfig,
    #     metadata={"help": "support for orignal args_dict"}
    # )

    num_inference_per_prompt: int = field(
        default=32,
        metadata={"help": "number of response for per prompt"}
    )
    seq_length: int = field(
        default=2048,
        metadata={"help": "length of prompt + response"}
    )
    max_new_tokens: int = field(
        default=2048,
        metadata={"help": "length of response"}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for sample train data"}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "top_p for sample train data"}
    )
    top_k: int = field(
        default=-1,
        metadata={"help": "top_k for sample train data"}
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "presence_penalty for sample train data"}
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "frequency_penalty for sample train data"}
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "repetition_penalty for sample train data"}
    )

    eval_temperature: float = field(
        default=0.6,
        metadata={"help": "temperature for sample eval data"}
    )
    eval_top_p: float = field(
        default=0.95,
        metadata={"help": "top_p for sample eval data"}
    )
    eval_top_k: int = field(
        default=20,
        metadata={"help": "top_k for sample eval data"}
    )
    eval_presence_penalty: float = field(
        default=0.0,
        metadata={"help": "presence_penalty for sample eval data"}
    )
    eval_frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "frequency_penalty for sample eval data"}
    )
    eval_repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "repetition_penalty for sample eval data"}
    )
    vllm_prompt_key: str = field(
        default="prompt",
        metadata={"help": "vllm_prompt_key"}
    )
    vllm_input_ids_key: str = field(
        default="input_ids",
        metadata={"help": "vllm_input_ids_key"}
    )
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "whether enable think or not"}
    )
    max_num_batched_tokens: int = field(
        default=32768,
        metadata={"help": "max_num_batched_tokens"}
    )
    max_seq_len_to_capture: int = field(
        default=2348,
        metadata={"help": "max_seq_len_to_capture"}
    )
    enable_stage_resume: bool = field(
        default=False,
        metadata={"help": "enable_stage_resume"}
    )
    gpu_memory_utilization: float = field(
        default=0.8,
        metadata={"help": "gpu_memory_utilization"}
    )
    enforce_eager: bool = field(
        default=False,
        metadata={"help": "enforce_eager"}
    )
    tokenizer: str = field(
        default=MISSING,
        metadata={"help": "tokenizer config path"}
    )
    seed: int = field(
        default=1234,
        metadata={"help": "seed"}
    )

@dataclass
class RefPolicyArgsDictConfig(BaseConfig):
    # seed: int = field(
    #     default=1234,
    #     metadata={"help": "seed"}
    # )
    pretrain_or_model: str = field(
        default=MISSING,
        metadata={"help": "path to reference model"}
    )

@dataclass
class RefPolicyConfig(BaseModelConfig):
    """RefPolicyConfig"""

    # args_dict: RefPolicyArgsDictConfig = field(
    #     default_factory=RefPolicyArgsDictConfig,
    #     metadata={"help": "support for orignal args_dict"}
    # )

    pretrain_or_model: str = field(
        default=MISSING,
        metadata={"help": "path to reference model"}
    )

@dataclass
class PolicyTrainerArgsDictConfig(BaseConfig):
    pretrain_or_model: str = field(
        default=MISSING,
        metadata={"help": "path to reference model"}
    )
    learning_rate: float = field(
        default=2e-06,
        metadata={"help": "learning rate for policy model"}
    )
    grad_clip: float = field(
        default=1.0,
        metadata={"help": "grad clips for policy model"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "whether gradient checkpointing"}
    )
    pos_clip_ratio: float = field(
        default=0.2
    )
    negative_clip_ratio: float = field(
        default=0.2
    )
    save_hf: bool = field(
        default=True
    )

@dataclass
class PolicyTrainerConfig(BaseModelConfig):
    """PolicyTrainerConfig"""

    # args_dict: PolicyTrainerArgsDictConfig = field(
    #     default_factory=PolicyTrainerArgsDictConfig,
    #     metadata={"help": "support for orignal args_dict"}
    # )

    pretrain_or_model: str = field(
        default=MISSING,
        metadata={"help": "path to reference model"}
    )
    learning_rate: float = field(
        default=2e-06,
        metadata={"help": "learning rate for policy model"}
    )
    grad_clip: float = field(
        default=1.0,
        metadata={"help": "grad clips for policy model"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "whether gradient checkpointing"}
    )
    pos_clip_ratio: float = field(
        default=0.2
    )
    negative_clip_ratio: float = field(
        default=0.2
    )
    save_hf: bool = field(
        default=True
    )


@dataclass
class RuntimeConfig(BaseConfig):
    """RuntimeConfig"""
    # setup config
    exp_name: str = field(
        default="CHATLEARN",
        metadata={"help": "exp name for each run"}
    )
    colocation: List[str] = field(
        default_factory=list,
        metadata={"help": "colocate models into the same device"}
    )
    concurrent_setup: bool = field(
        default=False,
        metadata={"help": "whether concurrent model setup or not"}
    )
    debug: bool = field(
        default=False,
        metadata={"help": "whether log debug infromation or not"}
    )
    nsys: bool = field(
        default=False,
        metadata={"help": "whether enable nsys nvtx"}
    )

    # path config
    output_dir: str = field(
        default=MISSING,
        metadata={"help": "output dir"}
    )
    profiler_dir: Optional[str] = field(
        default=None,
        metadata={"help": "profiler dir"}
    )
    data_path: str = field(
        default=MISSING,
        metadata={"help": "[required]: data_path for dataset or a List of data_path for different kind of datasets split by ,"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the evaluation data file."}
    )
    data_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "[optional]: checkpoint for dataloader"}
    )

    # config for training
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
    enable_resume_training: bool = field(
        default=True,
        metadata={"help": "[optional]: enable resume training when data checkpoint is set"}
    )

    # config for data
    data_ratio: List[int] = field(
        default_factory=list,
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
    max_replay_episode: int = field(
        default=0,
        metadata={"help": "max number of replay episodes, if `max_replay_episode` is set to -1, then replay all episodes \
                        if `max_replay_episode` is set to 0, then replay is disabled"}
    )
    replay_episode_offset: int = field(
        default=0,
        metadata={"help": "replay after n episodes"}
    )
    consumed_samples: int = field(
        default=0,
        metadata={"help": "consumed samples"}
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

    # eval config
    eval_episode_interval: int = field(
        default=0,
        metadata={"help": "[optional]: eval every N episode, if 0, will not eval"}
    )
    enable_eval_before_training: bool = field(
        default=False,
        metadata={"help": "whether to eval before training"}
    )

    # param sync config
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
    bucket_size_mb_in_memory_manager: int = field(
        default=1024,
        metadata={"help": "bucket size in the memory manager to reduce peak memory"}
    )
    free_sync_collective_group: bool = field(
        default=False,
        metadata={"help": "free collective group after parameter synchronization and rebuild before next synchronization"}
    )
    cpu_schedule_strategy: str = field(
        default=RAY_PG_STRATEGY.SPREAD.value,
        metadata={"help": "[optional] cpu only model schedule policy, PACK or SPREAD \
                    PACK: All provided bundles are packed onto a single node on a best-effort basis. \
                    SPREAD: Each bundle is spread onto separate nodes on a best-effort basis."}
    )
    validate_param_sync: bool = field(
        default=False,
        metadata={"help": "validate param sync"}
    )

    # graph config
    policy_to_regroup_queue: str = field(
        default="global_barrier",
        metadata={"help": "policy to regroup queue"}
    )

    # log config
    log_args_dict: LogConfig = field(
        default_factory=LogConfig,
        metadata={"help": "config for logging"}
    )
    log_interval: int = field(
        default=1,
        metadata={"help": "[optional] log time and memory per `log_interval` iterations."}
    )
