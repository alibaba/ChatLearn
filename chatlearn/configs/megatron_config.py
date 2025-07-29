# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
from typing import Optional
from dataclasses import dataclass, field

from omegaconf import MISSING

from megatron.training.arguments import moe_freq_type
from megatron.core.transformer.enums import AttnBackend

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
    num_query_groups: Optional[int] = field(
        default=None, metadata={"help": "num query groups"}
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
        default=False, metadata={"help": "Enable bias only in the QKV linear layers"}
    )
    add_bias_linear: bool = field(
        default=False, metadata={"help": "Whether bias in the linear layers"}
    )
    rotary_base: int = field(
        default=1000000,
        metadata={"help": "Base to use for rotary positional embeddings"},
    )
    group_query_attention: bool = field(
        default=False, metadata={"help": "Use group-query attention."}
    )
    untie_embeddings_and_output_weights: bool = field(
        default=False, metadata={"help": "Untie embeddings and output weights."}
    )
    qk_layernorm: bool = field(
        default=False, metadata={"help": "qk layer norm"}
    )
    vocab_size: int = field(default=32000)
    kv_channels: int = field(
        default=MISSING, metadata={"help": "kv channels"}
    )
    num_experts: Optional[int] = field(
        default=None,
        metadata={"help": "Number of Experts in MoE (None means no MoE)"},
    )
    moe_grouped_gemm: bool = field(
        default=False, metadata={"help": "When there are multiple experts per rank, compress multiple local (potentially small) gemms \
    in a single kernel launch to improve the utilization and performance by leveraging the Grouped \
    GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm)."}
    )
    moe_token_dispatcher_type: str = field(
        default="allgather",
        metadata={"help": "The type of token dispatcher to use. The default is 'allgather'."},
    )
    moe_router_topk: int = field(
        default=2,
        metadata={"help": "Number of experts to route to for each token. The default is 2."},
    )
    moe_router_load_balancing_type: str = field(
        default="aux_loss",
        metadata={"help": "Determines the load balancing strategy for the router."},
    )
    moe_router_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Data type for routing and expert output weighted averaging. Using fp32 or fp64 can \
    improve stability especially when the number of experts is large"},
    )
    moe_router_pre_softmax: bool = field(
        default=False, metadata={"help": "Enable pre-softmax(pre-sigmoid) routing for MoE, which means softmax is before the \
    top-k selection."}
    )
    moe_layer_freq: moe_freq_type = field(
        default=1,
        metadata={"help": "Frequency between MoE layers and Dense layers. Accepts either: \
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers. \
    - A list that defines a custom pattern, e.g.: [1,1,1,0,1,1,1,0,1,1,1,0]"},
    )
    moe_ffn_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "The hidden size of each expert\'s feed-forward network (ffn). "},
    )
    moe_aux_loss_coeff: float = field(
        default=0.0, metadata={"help": "Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended."}
    )
    q_lora_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Rank of Key and Value tensors' low rank representation."},
    )
    kv_lora_rank: int = field(
        default=32,
        metadata={"help": "Rank of Key and Value tensors' low rank representation."},
    )
    moe_router_num_groups: Optional[int] = field(
        default=None,
        metadata={"help": "Number of groups to divide experts into for group-limited routing"},
    )
    moe_router_group_topk: Optional[int] = field(
        default=None,
        metadata={"help": "Number of selected groups for group-limited routing."},
    )
    moe_shared_expert_intermediate_size: Optional[int] = field(
        default=None,
        metadata={"help": "Shared expert total ffn hidden size. "},
    )
    moe_router_score_function: str = field(
        default="softmax",
        metadata={"help": "Score function for MoE TopK routing. Can be softmax or sigmoid."},
    )
    moe_router_enable_expert_bias: bool = field(
        default=False, metadata={"help": "TopK routing with dynamic expert bias in the aux-loss-free load balancing strategy."}
    )
    multi_latent_attention: bool = field(
        default=False, metadata={"help": "Use multi-latent attention for model."}
    )
    v_head_dim: Optional[int] = field(
        default=None,
        metadata={"help": "v_head_dim "},
    )
    moe_router_topk_scaling_factor: float = field(
        default=0.0, metadata={"help": "Scaling factor for routing score in top-k selection, only works when --moe-router-pre-softmax enabled. "}
    )
    moe_router_pre_softmax: bool = field(
        default=False, metadata={"help": "Use multi-latent attention for model."}
    )
    apply_rope_fusion: bool = field(
        default=True, metadata={"help": "Disable rope fusion, the fusion is available"}
    )

    disable_bf16_reduced_precision_matmul: bool = field(
        default=False, metadata={"help": "prevent matmul from using reduced precision accumulation when using BF16"}
    )
    moe_shared_expert_overlap: bool = field(
        default=False, metadata={"help": "Enable overlapping between shared expert computations and dispatcher communications."}
    )
    moe_router_load_balancing_type: str = field(
        default="aux_loss",
        metadata={"help": "moe_router_load_balancing_type"},
    )
    bias_swiglu_fusion: bool = field(
        default=False, metadata={"help": "Disable swiglu fusion, the fusion is available"}
    )
    bias_dropout_fusion: bool = field(
        default=False, metadata={"help": "Disable dropout fusion, the fusion is available"}
    )
    rotary_scaling_factor: float = field(
        default=1.0, metadata={"help": "rotary_scaling_factor "}
    )


@dataclass
class MegatronTrainConfig(BaseConfig):
    expert_tensor_parallel_size: Optional[int] = field(
        default=None, metadata={"help": "expert tensor parallel size for Megatron-Core"}
    )
    decoder_first_pipeline_num_layers: Optional[int] = field(
        default=None, metadata={"help": "The number of transformer layers on the first pipeline stage of the decoder."}
    )
    decoder_last_pipeline_num_layers: Optional[int] = field(
        default=None, metadata={"help": "The number of transformer layers on the last pipeline stage of the decoder."}
    )
    load: str = field(default=MISSING, metadata={"help": "path to train model"})
    save: str = field(default=MISSING, metadata={"help": "path to save model"})
    tokenizer_type: str = field(
        default="NullTokenizer", metadata={"help": "What type of tokenizer to use."}
    )
    tokenizer_model: Optional[str] = field(
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
    log_interval: int = field(
        default=1,
        metadata={
            "help": "[optional] log time and memory per `log_interval` iterations."
        },
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
    recompute_granularity: Optional[str] = field(
        default=None,
        metadata={
            "help": "Checkpoint activations to allow for training with larger models, sequences, and batch sizes. \
            It is supported at two granularities 1) full: whole transformer layer is recomputed, \
            2) selective: submodules set in --recompute-modules are recomputed, default is core_attn."
        },
    )
    sequence_parallel: bool = field(
        default=False,
        metadata={"help": "Enable sequence parallel optimization for mcore"},
    )
    use_distributed_optimizer: bool = field(
        default=False,
        metadata={"help": "use_distributed_optimizer"},
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
    moe_router_force_load_balancing: bool = field(
        default=False, metadata={"help": "Force load balancing with random logits for MoE router, supports naive topk \
    and group-limited topk."}
    )
    gradient_accumulation_fusion: bool = field(
        default=False, metadata={"help": "If true, fuses weight gradient accumulation to GEMMs. Requires the custom CUDA extension \
            fused_weight_gradient_mlp_cuda module.."}
    )
    async_tensor_model_parallel_allreduce: bool = field(
        default=False, metadata={"help": "async_tensor_model_parallel_allreduce."}
    )
    overlap_p2p_comm: bool = field(
        default=False, metadata={"help": "When True some of the peer to peer communication for pipeline parallelism will overlap with computation。"}
    )
    batch_p2p_comm: bool = field(
        default=False, metadata={"help": "Use batch_isend_irecv instead of individual isend/irecv calls."}
    )
    deallocate_pipeline_outputs: bool = field(
        default=False, metadata={"help": "If True, output data is deallocated after the tensor is sent to the next pipeline stage."}
    )
    attention_backend: lambda attn_backend: AttnBackend[attn_backend] = field(
        default=AttnBackend.auto, metadata={"help": "Attention backend to use (flash,fused,unfused,local,auto). Defaults to auto"}
    )
    attention_softmax_in_fp32: bool = field(
        default=True, metadata={"help": "attention_softmax_in_fp32."}
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
    sequence_parallel: bool = field(
        default=True,
        metadata={"help": "Enable sequence parallel optimization for mcore"},
    )
    seq_length: int = field(
        default=MISSING,
        metadata={"help": "Maximum sequence length to process."},
    )
    tokenizer_type: str = field(
        default="NullTokenizer", metadata={"help": "What type of tokenizer to use."}
    )
    tokenizer_model: Optional[str] = field(
        default=None, metadata={"help": "pretrained model name or path"}
    )
    bf16: bool = field(default=True, metadata={"help": "Run model in bfloat16 mode."})
    expert_tensor_parallel_size: Optional[int] = field(
        default=None, metadata={"help": "expert tensor parallel size for Megatron-Core"}
    )
    decoder_first_pipeline_num_layers: Optional[int] = field(
        default=None, metadata={"help": "The number of transformer layers on the first pipeline stage of the decoder."}
    )
    decoder_last_pipeline_num_layers: Optional[int] = field(
        default=None, metadata={"help": "The number of transformer layers on the last pipeline stage of the decoder."}
    )
    moe_router_force_load_balancing: bool = field(
        default=False, metadata={"help": "moe_router_force_load_balancing."}
    )


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
