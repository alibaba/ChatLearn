# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
from dataclasses import dataclass, field

from omegaconf import MISSING

from chatlearn.configs.common import BaseConfig, BaseModelConfig, OptimizerConfig


@dataclass
class MegatronModelArchitectureConfig(BaseConfig):
    attention_dropout: float = field(
        default=0.0, metadata={"help": "Post attention dropout probability."}
    )
    hidden_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability for hidden state transformer."},
    )
    num_layers: int = field(
        default=MISSING, metadata={"help": "Number of transformer layers."}
    )
    hidden_size: int = field(
        default=MISSING, metadata={"help": "Tansformer hidden size."}
    )
    num_attention_heads: int = field(
        default=MISSING, metadata={"help": "Number of transformer attention heads."}
    )
    ffn_hidden_size: int = field(
        default=MISSING,
        metadata={"help": "Transformer Feed-Forward Network hidden size. "},
    )
    num_query_groups: int = field(
        default=MISSING, metadata={"help": "num query groups"}
    )
    seq_length: int = field(
        default=MISSING, metadata={"help": "Maximum sequence length to process."}
    )
    max_position_embeddings: int = field(
        default=MISSING,
        metadata={"help": "Maximum number of position embeddings to use. "},
    )
    swiglu: bool = field(
        default=True,
        metadata={
            "help": "Use gated linear units and SiLU activation instead of default gelu"
        },
    )
    normalization: str = field(
        default="RMSNorm",
        metadata={"help": "Which normalization technique to use. LayerNorm or RMSNorm"},
    )
    norm_epsilon: float = field(
        default=1e-6, metadata={"help": "Epsilon for layer norm and RMS norm."}
    )
    position_embedding_type: str = field(
        default="rope", metadata={"help": "Position embedding type."}
    )
    add_qkv_bias: bool = field(
        default=MISSING, metadata={"help": "Enable bias only in the QKV linear layers"}
    )
    add_bias_linear: bool = field(
        default=MISSING, metadata={"help": "Whether bias in the linear layers"}
    )
    rotary_base: int = field(
        default=1000000,
        metadata={"help": "Base to use for rotary positional embeddings"},
    )
    group_query_attention: bool = field(
        default=MISSING, metadata={"help": "Use group-query attention."}
    )
    untie_embeddings_and_output_weights: bool = field(
        default=MISSING, metadata={"help": "Untie embeddings and output weights."}
    )
    tokenizer_type: str = field(
        default="NullTokenizer", metadata={"help": "What type of tokenizer to use."}
    )
    patch_tokenizer_type: str = field(
        default=MISSING,
        metadata={"help": "What type of tokenizer to use. should match true model"},
    )
    vocab_size: int = field(default=32000)
    extra_vocab_size: int = field(
        default=MISSING, metadata={"help": "config vocab_size - tokenizer vocab_size"}
    )


@dataclass
class MegatronTrainConfig(BaseConfig):
    load: str = field(default=MISSING, metadata={"help": "path to train model"})
    save: str = field(default=MISSING, metadata={"help": "path to save model"})
    save_interval: int = field(
        default=MISSING,
        metadata={"help": "Number of iterations between persistent checkpoint saves."},
    )
    train_iters: int = field(
        default=MISSING,
        metadata={"help": "Total number of iterations to train over all"},
    )
    bf16: bool = field(default=True, metadata={"help": "Run model in bfloat16 mode."})
    use_checkpoint_opt_param_scheduler: bool = field(
        default=True,
        metadata={
            "help": "Use checkpoint to set the values of the scheduler \
                       (learning rate, warmup iterations, minimum learning \
                       rate, maximum number of iterations, and decay style \
                       from checkpoint and ignore input arguments."
        },
    )
    recompute_granularity: str = field(
        default="selective",
        metadata={
            "help": "Checkpoint activations to allow for training with larger models, sequences, and batch sizes. \
            It is supported at two granularities 1) full: whole transformer layer is recomputed, \
            2) selective: submodules set in --recompute-modules are recomputed, default is core_attn."
        },
    )
    sequence_parallel: bool = field(
        default=True,
        metadata={"help": "Enable sequence parallel optimization for mcore"},
    )
    no_load_optim: bool = field(
        default=True,
        metadata={"help": "Do not load optimizer when loading checkpoint."},
    )
    no_load_rng: bool = field(
        default=True,
        metadata={"help": "Do not load rng state when loading checkpoint."},
    )
    no_load_scheduler: bool = field(
        default=True,
        metadata={"help": "Do not load scheduler when loading checkpoint."},
    )
    finetune: bool = field(
        default=True,
        metadata={
            "help": "Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0."
        },
    )
    optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig, metadata={"help": "optimizer config"}
    )


@dataclass
class MegatronRefPolicyConfig(BaseModelConfig):
    """RefPolicyConfig"""
    megatron_model_cfg: MegatronModelArchitectureConfig = field(
        default_factory=MegatronModelArchitectureConfig,
        metadata={"help": "cfg for megatron model architecture, should in megatron's arguments"}
    )

    load: str = field(default=MISSING, metadata={"help": "path to reference model"})
    seed: int = field(default=1234, metadata={"help": "seed"})


@dataclass
class MegatronPolicyTrainerConfig(
    BaseModelConfig, MegatronTrainConfig
):
    """PolicyTrainerConfig"""
    megatron_model_cfg: MegatronModelArchitectureConfig = field(
        default_factory=MegatronModelArchitectureConfig,
        metadata={"help": "cfg for megatron model architecture, should in megatron's arguments"}
    )
    pos_clip_ratio: float = field(default=0.2)
    neg_clip_ratio: float = field(default=0.2)
    diff_clip_ratio: float = field(default=10)
    final_clip_ratio: float = field(default=3)
    seed: int = field(default=1234, metadata={"help": "seed"})
