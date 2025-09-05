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
"""megatron_utils"""

from megatron.core.transformer.enums import AttnBackend
from transformers import AutoConfig

def update_cfg(cfg):

    hf_transformer_config = AutoConfig.from_pretrained(cfg.models.policy.load)

    # common cfgs
    cfg.models.policy_trainer.megatron_model_cfg.attention_dropout = hf_transformer_config.attention_dropout
    cfg.models.policy_trainer.megatron_model_cfg.num_layers = hf_transformer_config.num_hidden_layers
    cfg.models.policy_trainer.megatron_model_cfg.hidden_size = hf_transformer_config.hidden_size
    cfg.models.policy_trainer.megatron_model_cfg.num_attention_heads = hf_transformer_config.num_attention_heads
    cfg.models.policy_trainer.megatron_model_cfg.ffn_hidden_size = hf_transformer_config.intermediate_size
    cfg.models.policy_trainer.megatron_model_cfg.max_position_embeddings = hf_transformer_config.max_position_embeddings
    cfg.models.policy_trainer.megatron_model_cfg.add_qkv_bias = False
    cfg.models.policy_trainer.megatron_model_cfg.add_bias_linear = False
    cfg.models.policy_trainer.megatron_model_cfg.rotary_base = hf_transformer_config.rope_theta
    cfg.models.policy_trainer.megatron_model_cfg.norm_epsilon = hf_transformer_config.rms_norm_eps
    cfg.models.policy_trainer.megatron_model_cfg.untie_embeddings_and_output_weights = not hf_transformer_config.tie_word_embeddings
    cfg.models.policy_trainer.megatron_model_cfg.vocab_size = hf_transformer_config.vocab_size
    cfg.models.policy_trainer.megatron_model_cfg.qk_layernorm = True
    cfg.models.policy_trainer.megatron_model_cfg.kv_channels = hf_transformer_config.head_dim

    if "Qwen" in hf_transformer_config.architectures[0]:
        cfg.models.policy_trainer.megatron_model_cfg.group_query_attention = True
        cfg.models.policy_trainer.megatron_model_cfg.num_query_groups = hf_transformer_config.num_key_value_heads

    # moe cfgs
    if hasattr(hf_transformer_config, 'moe_intermediate_size'):
        cfg.models.policy_trainer.megatron_model_cfg.moe_grouped_gemm = True
        cfg.models.policy_trainer.megatron_model_cfg.moe_token_dispatcher_type = "alltoall"
        cfg.models.policy_trainer.megatron_model_cfg.moe_router_topk = hf_transformer_config.num_experts_per_tok
        cfg.models.policy_trainer.megatron_model_cfg.moe_ffn_hidden_size = hf_transformer_config.moe_intermediate_size
        cfg.models.policy_trainer.megatron_model_cfg.moe_router_dtype= 'fp64'

        if "Qwen3MoeForCausalLM" == hf_transformer_config.architectures[0]:
            cfg.models.policy_trainer.megatron_model_cfg.num_experts = hf_transformer_config.num_experts
            cfg.models.policy_trainer.megatron_model_cfg.moe_layer_freq = [1] * hf_transformer_config.num_hidden_layers
        elif "DeepseekV3ForCausalLM" == hf_transformer_config.architectures[0]:
            cfg.models.policy_trainer.megatron_model_cfg.num_experts = hf_transformer_config.n_routed_experts
            cfg.models.policy_trainer.megatron_model_cfg.moe_layer_freq = [0] * hf_transformer_config.first_k_dense_replace \
                + [1] * (hf_transformer_config.num_hidden_layers - hf_transformer_config.first_k_dense_replace)
            cfg.models.policy_trainer.megatron_model_cfg.q_lora_rank = hf_transformer_config.q_lora_rank
            cfg.models.policy_trainer.megatron_model_cfg.kv_lora_rank = hf_transformer_config.kv_lora_rank
            cfg.models.policy_trainer.megatron_model_cfg.moe_shared_expert_intermediate_size = hf_transformer_config.n_shared_experts \
                 * hf_transformer_config.moe_intermediate_size
            cfg.models.policy_trainer.megatron_model_cfg.moe_router_score_function = "sigmoid"
            cfg.models.policy_trainer.megatron_model_cfg.moe_router_enable_expert_bias = True
            cfg.models.policy_trainer.megatron_model_cfg.multi_latent_attention = True
            cfg.models.policy_trainer.megatron_model_cfg.v_head_dim = hf_transformer_config.v_head_dim
            cfg.models.policy_trainer.megatron_model_cfg.moe_router_topk_scaling_factor = hf_transformer_config.routed_scaling_factor
            cfg.models.policy_trainer.megatron_model_cfg.moe_router_pre_softmax = True
            cfg.models.policy_trainer.megatron_model_cfg.apply_rope_fusion = False
            cfg.models.policy_trainer.megatron_model_cfg.disable_bf16_reduced_precision_matmul = True
            cfg.models.policy_trainer.megatron_model_cfg.moe_shared_expert_overlap = True
            cfg.models.policy_trainer.megatron_model_cfg.moe_router_load_balancing_type = "seq_aux_loss"
            cfg.models.policy_trainer.megatron_model_cfg.moe_aux_loss_coeff = 0.001
            cfg.models.policy_trainer.megatron_model_cfg.bias_swiglu_fusion = True
            cfg.models.policy_trainer.megatron_model_cfg.bias_dropout_fusion = True
            cfg.models.policy_trainer.megatron_model_cfg.gradient_accumulation_fusion = False
            cfg.models.policy_trainer.megatron_model_cfg.async_tensor_model_parallel_allreduce = False
            cfg.models.policy_trainer.megatron_model_cfg.overlap_p2p_comm = True
            cfg.models.policy_trainer.megatron_model_cfg.batch_p2p_comm = False
            cfg.models.policy_trainer.megatron_model_cfg.deallocate_pipeline_outputs = False
            cfg.models.policy_trainer.megatron_model_cfg.attention_backend = AttnBackend.auto
            cfg.models.policy_trainer.megatron_model_cfg.attention_softmax_in_fp32 = True
            cfg.models.policy_trainer.megatron_model_cfg.rotary_scaling_factor = 40
    if cfg.models.policy_trainer.context_parallel_size > 1:
        cfg.models.policy_trainer.megatron_model_cfg.apply_rope_fusion = False
    cfg.models.ref_policy.megatron_model_cfg = cfg.models.policy_trainer.megatron_model_cfg

    return cfg
