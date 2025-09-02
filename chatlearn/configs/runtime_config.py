"""Configs for runtime"""
from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import MISSING

from chatlearn.utils.constant import RAY_PG_STRATEGY
from .base import BaseConfig


__all__ = ['RuntimeConfig', 'RuntimeEnvConfig']

@dataclass
class LogConfig(BaseConfig):
    """Configs for ChatLearn Logging"""
    log_dir: str = field(default=MISSING, metadata={"help": "log dir"})
    # config for tensorboard
    enable_tensorboard: bool = field(
        default=False, metadata={"help": "whether enable tensorboard or not"}
    )
    tensorboard_dir: Optional[str] = field(
        default=None, metadata={"help": "tensorboard file save dir"}
    )
    # config for wandb
    enable_wandb: bool = field(
        default=False, metadata={"help": "whether enable wandb or not"}
    )
    wandb_dir: Optional[str] = field(
        default=None, metadata={"help": "wandb file save dir"}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "wandb project"}
    )
    wandb_id: Optional[str] = field(default=None, metadata={"help": "wandb id"})
    wandb_name: Optional[str] = field(default=None, metadata={"help": "wandb name"})
    wandb_resume: str = field(default="allow", metadata={"help": "wandb resume"})

    def _validate_impl(self):
        if self.enable_tensorboard:
            if self.tensorboard_dir is None:
                self.tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
        if self.enable_wandb:
            if self.wandb_dir is None:
                self.wandb_dir = os.path.join(self.log_dir, 'wandb')
        # TODO: check wandb_id, wandb_name and wandb_resume


@dataclass
class RuntimeConfig(BaseConfig):
    """RuntimeConfig"""

    # setup config
    train_backend: str = field(
        default=MISSING,
        metadata={"help": "which train backend to use, one of megatron or fsdp"},
    )
    rollout_backend: str = field(
        default="vllm", metadata={"help": "rollout backend type, one of vllm or sglang"}
    )
    exp_name: str = field(
        default="CHATLEARN", metadata={"help": "exp name for each run"}
    )
    colocation: List[str] = field(
        default_factory=list, metadata={"help": "colocate models into the same device"}
    )
    concurrent_setup: bool = field(
        default=False, metadata={"help": "whether concurrent model setup or not"}
    )
    debug: bool = field(
        default=False, metadata={"help": "whether log debug infromation or not"}
    )
    nsys: bool = field(default=False, metadata={"help": "whether enable nsys nvtx"})

    # path config
    output_dir: str = field(default=MISSING, metadata={"help": "output dir"})
    profiler_dir: Optional[str] = field(default=None, metadata={"help": "profiler dir"})
    data_path: str = field(
        default=MISSING,
        metadata={
            "help": "[required]: data_path for dataset or a List of data_path for different kind of datasets split by ,"
        },
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data file."}
    )
    data_checkpoint_path: Optional[str] = field(
        default=None, metadata={"help": "[optional]: checkpoint for dataloader"}
    )

    # config for training
    num_episode: int = field(
        default=MISSING,
        metadata={
            "help": "[required] number of episodes. One episode includes a inference and training loop."
        },
    )
    sample_per_episode: int = field(
        default=MISSING, metadata={"help": "[required] number of samples per episode."}
    )
    num_training_epoch: int = field(
        default=1,
        metadata={
            "help": "[optional] number of training epoch per episode. default set to 1."
        },
    )
    train_micro_batch_size: int = field(
        default=MISSING, metadata={"help": "[required] training micro batch size."}
    )
    train_global_batch_size: int = field(
        default=MISSING, metadata={"help": "[required] training global batch size."}
    )
    save_episode_interval: int = field(
        default=MISSING,
        metadata={"help": "save checkpoint per `save_episode_interval` episodes."},
    )
    enable_resume_training: bool = field(
        default=True,
        metadata={
            "help": "[optional]: enable resume training when data checkpoint is set"
        },
    )

    # config for data
    data_ratio: List[int] = field(
        default_factory=list,
        metadata={"help": "the ratio for each kind of data_path in a training episode"},
    )
    data_shuffle: bool = field(
        default=True,
        metadata={"help": "shuffle in each epoch of dataset, default: True"},
    )
    data_rerank: bool = field(
        default=True, metadata={"help": "rerank batch of data by row"}
    )
    max_replay_episode: int = field(
        default=0,
        metadata={
            "help": "max number of replay episodes, if `max_replay_episode` is set to -1, then replay all episodes \
                        if `max_replay_episode` is set to 0, then replay is disabled"
        },
    )
    replay_episode_offset: int = field(
        default=0, metadata={"help": "replay after n episodes"}
    )
    consumed_samples: int = field(default=0, metadata={"help": "consumed samples"})
    max_data_ckpt_nums: Optional[int] = field(
        default=None, metadata={"help": "[optional]: max data checkpoint nums"}
    )
    load_data_checkpoint_iteration: Optional[int] = field(
        default=None,
        metadata={"help": "[optional]: load data checkpoint from iteration"},
    )
    stream_data_loader_type: str = field(
        default="fixed", metadata={"help": "stream_data_loader type, fixed or dynamic"}
    )

    # eval config
    eval_episode_interval: int = field(
        default=0,
        metadata={"help": "[optional]: eval every N episode, if 0, will not eval"},
    )
    enable_eval_before_training: bool = field(
        default=False, metadata={"help": "whether to eval before training"}
    )

    # param sync config
    bucket_size_mb_in_memory_manager: int = field(
        default=1024,
        metadata={"help": "bucket size in the memory manager to reduce peak memory"},
    )
    cpu_schedule_strategy: str = field(
        default=RAY_PG_STRATEGY.SPREAD.value,
        metadata={
            "help": "[optional] cpu only model schedule policy, PACK or SPREAD \
                    PACK: All provided bundles are packed onto a single node on a best-effort basis. \
                    SPREAD: Each bundle is spread onto separate nodes on a best-effort basis."
        },
    )

    # graph config
    policy_to_regroup_queue: str = field(
        default="global_barrier", metadata={"help": "policy to regroup queue"}
    )

    # log config
    log_args_dict: LogConfig = field(
        default_factory=LogConfig, metadata={"help": "config for logging"}
    )

    def _validate_impl(self):
        """valid this config, recursively called in `validate`.
        Should raise Error if failed.
        """
        assert self.sample_per_episode % self.train_global_batch_size == 0, \
            "runtime_args.sample_per_episode must be divisible by runtime_args.train_global_batch_size"            

    def _post_init_impl(self):
        # TODO: Currently supports single-layer colocation, and need to support more complex scenarios in the future.
        if self.colocation and isinstance(self.colocation[0], str):
            self.colocation = [self.colocation]

@dataclass
class RuntimeEnvConfig(BaseConfig):
    """Runtime env config, you can refer https://docs.ray.io/en/latest/ray-core/handling-dependencies.html for more information."""

    excludes: List[str] = field(
        default_factory=lambda: ["*pt", "logs", "tensorboards", ".nfs*"],
        metadata={"help": "excludes files from packaging"},
    )
    platform: str = field(
        default="DLC", metadata={"help": "Platform to run the model. Default is DLC."}
    )
    pip: List[str] = field(
        default_factory=list, metadata={"help": "pip install packages"}
    )
    working_dir: Optional[str] = field(
        default=None, metadata={"help": "working directory"}
    )
    py_modules: List[str] = field(
        default_factory=list, metadata={"help": "python modules"}
    )
