# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
"""common configs"""

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, List, Optional
import warnings

from omegaconf import MISSING

from chatlearn.utils.constant import RAY_PG_STRATEGY

@dataclass
class BaseConfig:
    # TODO: Unifying Parameter Access Using dataclass approach
    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"{key} is not a valid field")

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

    def validate(self):
        return True


@dataclass
class FreeGpuMemoryConfig(BaseConfig):
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


@dataclass
class LogConfig(BaseConfig):
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
    tensor_model_parallel_size: int = field(
        default=1, metadata={"help": "tensor model parallel size"}
    )
    pipeline_model_parallel_size: int = field(
        default=1, metadata={"help": "pipeline model parallel size"}
    )
    expert_model_parallel_size: int = field(
        default=1, metadata={"help": "expert model parallel size for Megatron-Core"}
    )
    expert_tensor_parallel_size: Optional[int] = field(
        default=None, metadata={"help": "expert tensor parallel size for Megatron-Core"}
    )
    fsdp_size: int = field(default=1, metadata={"help": "FSDP parallel size"})
    ulysses_sequence_parallel_size: int = field(
        default=1,
        metadata={"help": "ulysses sequence parallel size used for fsdp train backend"},
    )
    packing: bool = field(default=False, metadata={"help": "Whether to use sequence packing"})
    max_token_in_packing: int = field(
        default=32768, metadata={"help": "max token in packing when packing is enabled"}
    )
    meta_init: bool = field(
        default=False, metadata={"help": "Whether to use meta init for FSDP. When using groupgemm, recommend enable meta init"}
    )
    groupgemm: bool = field(
        default=False, metadata={"help": "Whether to use groupgemm patch for moe, now only support qwen3moe model"}
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


@dataclass
class PolicyConfig(BaseModelConfig):
    """PolicyConfig"""

    num_inference_per_prompt: int = field(
        default=32, metadata={"help": "number of response for per prompt"}
    )
    seq_length: int = field(
        default=2048, metadata={"help": "length of prompt + response"}
    )
    max_new_tokens: int = field(default=2048, metadata={"help": "length of response"})
    temperature: float = field(
        default=1.0, metadata={"help": "temperature for sample train data"}
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
    vllm_prompt_key: str = field(default="prompt", metadata={"help": "vllm_prompt_key"})
    vllm_input_ids_key: str = field(
        default="input_ids", metadata={"help": "vllm_input_ids_key"}
    )
    enable_thinking: bool = field(
        default=False, metadata={"help": "whether enable think or not"}
    )
    max_num_batched_tokens: int = field(
        default=32768, metadata={"help": "max_num_batched_tokens"}
    )
    max_seq_len_to_capture: int = field(
        default=2348, metadata={"help": "max_seq_len_to_capture"}
    )
    enable_stage_resume: bool = field(
        default=False, metadata={"help": "enable_stage_resume"}
    )
    gpu_memory_utilization: float = field(
        default=0.8, metadata={"help": "gpu_memory_utilization"}
    )
    enforce_eager: bool = field(default=False, metadata={"help": "enforce_eager"})
    load: str = field(default=MISSING, metadata={"help": "model weights and tokenizer config path"})
    seed: int = field(default=1234, metadata={"help": "seed"})


@dataclass
class FSDPConfig(BaseConfig):
    use_expandable_segments: bool = field(
        default=False, metadata={"help": "Whether to use expandable_segments in PYTORCH_CUDA_ALLOC_CONF, \
            avoid big reseverd memory in ref and policy trainer worker, expandable_segments should be False \
            while in parameter sync for efficiency"}
    )

@dataclass
class RefPolicyConfig(BaseModelConfig, FSDPConfig):
    """RefPolicyConfig"""

    load: str = field(
        default=MISSING, metadata={"help": "path to reference model"}
    )


@dataclass
class PolicyTrainerConfig(BaseModelConfig, FSDPConfig):
    """PolicyTrainerConfig"""

    load: str = field(
        default=MISSING, metadata={"help": "path to policy model"}
    )
    optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig, metadata={"help": "optimizer config"}
    )
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "whether gradient checkpointing"}
    )
    entropy_coef: float = field(
        default=0.0, metadata={"help": "entropy regularization"}
    )
    kl_coef: float = field(
        default=0.0, metadata={"help": "kl regularization"}
    )
    pos_clip_ratio: float = field(default=0.2)
    neg_clip_ratio: float = field(default=0.2)
    save_hf: bool = field(default=True)


@dataclass
class RuntimeConfig(BaseConfig):
    """RuntimeConfig"""

    # setup config
    train_backend: str = field(
        default=MISSING,
        metadata={"help": "which train backend to use, one of megatron or fsdp"},
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
    log_interval: int = field(
        default=1,
        metadata={
            "help": "[optional] log time and memory per `log_interval` iterations."
        },
    )
def _config_validate(cfg):
    # Check batchsize compatibility
    sample_per_episode = cfg.runtime_args.sample_per_episode
    train_global_batch_size = cfg.runtime_args.train_global_batch_size
    assert sample_per_episode % train_global_batch_size == 0, \
        "runtime_args.sample_per_episode must be divisible by runtime_args.train_global_batch_size"

    # Check vllm compatibility
    vllm_num_inference_per_prompt = cfg.models.policy.num_inference_per_prompt
    policy_num_gpu = cfg.models.policy.num_gpu
    assert sample_per_episode % vllm_num_inference_per_prompt == 0, \
        "total_train_sample must be divisible by models.policy.num_inference_per_prompt"
    assert policy_num_gpu % cfg.models.policy.tensor_model_parallel_size == 0, \
        "models.policy.num_gpu must be divisible by tensor_model_parallel_size"
    vllm_dp_size = policy_num_gpu // cfg.models.policy.tensor_model_parallel_size
    assert sample_per_episode % vllm_dp_size == 0, (
        "vllm_dp_size = models.policy.num_gpu // models.policy.tensor_model_parallel_size, \
            runtime_args.sample_per_episode must be divisible by vllm_dp_size"
    )

    # Check trainer/ref policy compatibility
    if cfg.runtime_args.train_backend=='fsdp':
        # Check FSDP compatibility for policy_trainer
        trainer_gpu = cfg.models.policy_trainer.num_gpu
        train_sp_size = cfg.models.policy_trainer.ulysses_sequence_parallel_size
        assert trainer_gpu % train_sp_size == 0, \
            "models.policy_trainer.num_gpu must be divisible by models.policy_trainer.ulysses_sequence_parallel_size"
        train_fsdp_dp_size = trainer_gpu // train_sp_size
        train_micro_batch_size = cfg.runtime_args.train_micro_batch_size
        train_generation_batch_size = cfg.models.policy_trainer.generation_batch_size
        assert train_global_batch_size % (train_fsdp_dp_size * train_micro_batch_size) == 0, (
            "train_fsdp_dp_size = models.policy_trainer.num_gpu // models.policy_trainer.ulysses_sequence_parallel_size \
                runtime_args.train_global_batch_size must be divisible by train_fsdp_dp_size"
        )
        assert sample_per_episode % train_generation_batch_size == 0, (
            "runtime_args.sample_per_episode must be divisible by models.policy_trainer.generation_batch_size"
        )
        if cfg.models.policy_trainer.packing:
            if train_global_batch_size != train_fsdp_dp_size * train_micro_batch_size:
                warning_message = f"In order to maximum packing capacity, runtime_args.train_micro_batch_size should be set to \
                    runtime_args.train_global_batch_size / (models.policy_trainer.num_gpu // models.policy_trainer.ulysses_sequence_parallel_size) = \
                    {train_global_batch_size // train_fsdp_dp_size}"
                warnings.warn(warning_message)
            if sample_per_episode != train_fsdp_dp_size * train_generation_batch_size:
                warning_message = f"In order to maximum packing capacity, models.policy_trainer.generation_batch_size should be set to \
                    runtime_args.sample_per_episode / (models.policy_trainer.num_gpu // models.policy_trainer.ulysses_sequence_parallel_size) = \
                    {sample_per_episode // train_fsdp_dp_size}"
                warnings.warn(warning_message)

        # Check FSDP compatibility for ref_policy
        ref_num_gpu = cfg.models.ref_policy.num_gpu
        ref_sp_size = cfg.models.ref_policy.ulysses_sequence_parallel_size
        ref_fsdp_dp_size = ref_num_gpu // ref_sp_size
        ref_generation_batch_size = cfg.models.ref_policy.generation_batch_size
        assert sample_per_episode % ref_generation_batch_size == 0, (
            "runtime_args.sample_per_episode must be divisible by models.ref_policy.generation_batch_size"
        )
        if cfg.models.ref_policy.packing:
            if sample_per_episode != ref_fsdp_dp_size * ref_generation_batch_size:
                warning_message = f"In order to maximum packing capacity, models.ref_policy.generation_batch_size should be set to \
                    runtime_args.sample_per_episode // (models.ref_policy.num_gpu // models.ref_policy.ulysses_sequence_parallel_size) = \
                    {sample_per_episode // ref_fsdp_dp_size}"
                warnings.warn(warning_message)
                