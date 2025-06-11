from dataclasses import dataclass, field

from omegaconf import MISSING

from .common import BaseConfig, BaseModelConfig


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
    # debug
    make_vocab_size_divisible_by: int = field(default=128)


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
    clip_grad: float = field(
        default=1.0, metadata={"help": "Gradient clipping based on global L2 norm."}
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
    lr: float = field(default=2e-6, metadata={"help": "Initial learning rate."})
    min_lr: float = field(
        default=0, metadata={"help": "Minimum value for learning rate."}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay coefficient for L2 regularization."},
    )


@dataclass
class MegatronRefPolicyConfig(BaseModelConfig, MegatronModelArchitectureConfig):
    """RefPolicyConfig"""

    load: str = field(default=MISSING, metadata={"help": "path to reference model"})
    # hard code
    # micro_batch_size: int = field(
    #     default=1,
    #     metadata={"help": "[required] micro batch size."}
    # )
    # global_batch_size: int = field(
    #     default=1,
    #     metadata={"help": "[required] global_batch_size."}
    # )
    seed: int = field(default=1234, metadata={"help": "seed"})


@dataclass
class MegatronPolicyTrainerConfig(
    BaseModelConfig, MegatronModelArchitectureConfig, MegatronTrainConfig
):
    """PolicyTrainerConfig"""

    pos_clip_ratio: float = field(default=0.2)
    neg_clip_ratio: float = field(default=0.2)
    diff_clip_ratio: float = field(default=10)
    final_clip_ratio: float = field(default=3)

    # hard code
    # micro_batch_size: int = field(
    #     default=1,
    #     metadata={"help": "[required] micro batch size."}
    # )

    # global_batch_size: int = field(
    #     default=1,
    #     metadata={"help": "[required] global_batch_size."}
    # )
    seed: int = field(default=1234, metadata={"help": "seed"})
