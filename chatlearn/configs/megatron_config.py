# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
from dataclasses import dataclass, field

from omegaconf import MISSING

from chatlearn.configs.common import BaseConfig, BaseModelConfig, OptimizerConfig

from megatron.training.arguments import moe_freq_type


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
    qk_layernorm: bool = field(
        default=MISSING, metadata={"help": "qk layer norm"}
    )
    vocab_size: int = field(default=32000)
    num_experts: int = field(
        default=None,
        metadata={"help": "Number of Experts in MoE (None means no MoE)"},
    )
    moe_grouped_gemm: bool = field(
        default=False, metadata={"help": "When there are multiple experts per rank, launch multiple local GEMM kernels in multiple streams to improve the utilization and performance with GroupedLinear in TransformerEngine."}
    )
    moe_token_dispatcher_type: str = field(
        default="allgather",
        metadata={"help": "The type of token dispatcher to use. The default is 'allgather'. Options are 'allgather', 'alltoall' and 'alltoall_seq'. We recommend using 'alltoall' when applying expert parallelism. For more information, please refer to the documentation in core/moe/README."},
    )
    moe_router_topk: int = field(
        default=2,
        metadata={"help": "Number of experts to route to for each token. The default is 2."},
    )
    moe_router_load_balancing_type: str = field(
        default="aux_loss",
        metadata={"help": "Determines the load balancing strategy for the router."},
    )
    moe_layer_freq: moe_freq_type = field(
        default=1,
        metadata={"help": "Frequency between MoE layers and Dense layers."},
    )
    moe_ffn_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of each expert\'s feed-forward network (ffn). "},
    )
    moe_aux_loss_coeff: float = field(
        default=0.0, metadata={"help": "Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended."}
    )



@dataclass
class MegatronTrainConfig(BaseConfig):
    
    expert_tensor_parallel_size: int = field(
        default=None, metadata={"help": "expert model parallel size for Megatron-Core"}
    )
    load: str = field(default=MISSING, metadata={"help": "path to train model"})
    save: str = field(default=MISSING, metadata={"help": "path to save model"})
    tokenizer_type: str = field(
        default="NullTokenizer", metadata={"help": "What type of tokenizer to use."}
    )
    tokenizer_model: str = field(
        default=None, metadata={"help": "pretrained model name or path"}
    )
    seq_length: int = field(
        default=MISSING,
        metadata={"help": "Maximum sequence length to process."},
    )
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
    seq_length: int = field(
        default=MISSING,
        metadata={"help": "Maximum sequence length to process."},
    )
    tokenizer_type: str = field(
        default="NullTokenizer", metadata={"help": "What type of tokenizer to use."}
    )
    tokenizer_model: str = field(
        default=None, metadata={"help": "pretrained model name or path"}
    )
    bf16: bool = field(default=True, metadata={"help": "Run model in bfloat16 mode."})


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
