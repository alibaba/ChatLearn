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
"""vllm utils"""

import argparse
import os
import re

from datetime import timedelta

import torch
import torch.distributed
from vllm.distributed.parallel_state import init_world_group
from vllm.distributed import parallel_state as mpu
from vllm.distributed.parallel_state import initialize_model_parallel
from vllm.model_executor.model_loader.utils import get_model_architecture  as get_model_architecture_v2

from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion


# The simple map of names for "automated" rules.
megatron_to_transformers = {
    "self_attention.dense": ".self_attn.o_proj.",
    "mlp.dense_h_to_4h": ".mlp.gate_up_proj.",
    "mlp.dense_4h_to_h": ".mlp.down_proj.",
    "mlp.gate_up_proj": ".mlp.gate_up_proj.",
    "mlp.down_proj": ".mlp.down_proj.",
    "self_attention.rotary_emb":".self_attn.rotary_emb.inv_freq",
    "self_attention.query_key_value": ".self_attn.qkv_proj",
    "attention.query_key_value": ".self_attn.qkv_proj",
}

megatron_qwen_to_transformers = {
    "attention.attention_layernorm": ".attn.attention_layernorm.",
    "attention.dense": ".attn.c_proj.",
    "self_attention.dense": ".attn.c_proj.",
    "mlp.dense_h_to_4h": ".mlp.c_fc.",
    "mlp.w1": ".mlp.gate_up_proj.",
    "mlp.w2": ".mlp.gate_up_proj.",
    "mlp.dense_4h_to_h": ".mlp.c_proj.",
    "mlp.dense_layernorm": "mlp.dense_layernorm",
}


megatron_qwen2_to_transformers = {
    "attention.attention_layernorm": ".attn.attention_layernorm.",
    "attention.dense": ".attn.c_proj.",
    "self_attention.dense": ".self_attn.o_proj.",
    "mlp.dense_h_to_4h": ".mlp.gate_up_proj.",
    "mlp.w1": ".mlp.gate_up_proj.",
    "mlp.w2": ".mlp.gate_up_proj.",
    "mlp.dense_4h_to_h": ".mlp.down_proj.",
    "mlp.dense_layernorm": ".mlp.dense_layernorm.",
    "mlp.router.layer": ".mlp.gate.",
    "mlp.experts.dense_h_to_4h": ".mlp.experts.w13_weight",
    "mlp.experts.dense_4h_to_h": ".mlp.experts.w2_weight",
    "mlp.shared_experts.dense_h_to_4h": ".mlp.shared_expert.gate_up_proj.",
    "mlp.shared_experts.dense_4h_to_h": ".mlp.shared_expert.down_proj.",
    "mlp.gate": ".mlp.shared_expert_gate."
}


mcore_to_transformers = {
    "self_attention.linear_proj": ".self_attn.o_proj.",
    "mlp.linear_fc1": ".mlp.gate_up_proj.",
    "mlp.linear_fc2": ".mlp.down_proj.",
}

def get_model_architecture(config):
    return get_model_architecture_v2(config)[0]


def get_pipeline_model_parallel_rank():
    return mpu.get_pp_group().rank_in_group


def get_pipeline_model_parallel_world_size():
    return mpu.get_pp_group().world_size

class ParameterSyncMap:
    """Base ParameterSyncMap."""
    def __init__(self, src_names, layer_offset):
        self.weight_or_bias = ["weight", "bias"]
        self.src_names = src_names
        self.layer_offset = layer_offset
        self._dst_names = []

    @property
    def embedding_sync_map(self):
        return self._embedding_sync_map

    @property
    def layer_sync_map(self):
        return self._layer_sync_map

    @property
    def final_layer_sync_map(self):
        return self._final_layer_sync_map

    @property
    def concat_params_dict(self):
        return self._concat_params_dict

    @property
    def to_fix_shared_expert_ordering(self):
        return self._to_fix_shared_expert_ordering

    @property
    def to_allgather_roututed_experts_dict(self):
        return self._to_allgather_routed_experts_dict

    @property
    def to_alltoall_roututed_experts_dict(self):
        return self._to_alltoall_routed_experts_dict

    @property
    def to_fix_act_ordering_dict(self):
        return self._to_fix_act_ordering_dict

    @property
    def to_fix_qkv_ordering_dict(self):
        return self._to_fix_qkv_ordering_dict

    @property
    def dst_names(self):
        if not self._dst_names:
            self.map_src_to_dst()
        return self._dst_names

    def map_src_to_dst(self):
        raise RuntimeError("Must be implemented by subclass.")

    def get_dst_name(self, sync_map, src_name):
        assert src_name in sync_map, f"expect {src_name} in {sync_map}"
        return sync_map[src_name]


class MCore2LlamaSyncMap(ParameterSyncMap):
    """sync map:megatron-core to llama transformer"""
    def __init__(self, src_names, layer_offset):
        src_prefix = "module.module"
        dst_prefix = "model"
        # The regex to extract layer names.
        self.layer_re = re.compile(rf"{src_prefix}.decoder.layers\.(\d+)\.([a-z0-9_.]+)[\._]([a-z]+)")
        self.src_prefix = src_prefix
        self.dst_prefix = dst_prefix
        # vLLM skips loading rotary_pos_emb and re-initializes it. Thus, we don't synchronize it from MCore to vllm.
        self._embedding_sync_map = {
            f"{src_prefix}.embedding.word_embeddings.weight": f"{dst_prefix}.embed_tokens.weight",
        }
        self._layer_sync_map = {
            "self_attention.linear_proj": ".self_attn.o_proj.",
            "mlp.linear_fc1": ".mlp.gate_up_proj.",
            "mlp.linear_fc2": ".mlp.down_proj.",
        }
        self._final_layer_sync_map = {
            f"{src_prefix}.decoder.final_layernorm.weight": f"{dst_prefix}.norm.weight",
            f"{src_prefix}.output_layer.weight": "lm_head.weight"
        }
        self._concat_params_dict = None
        self._to_fix_shared_expert_ordering = None
        self._to_allgather_routed_experts_dict = None
        self._to_alltoall_routed_experts_dict = None
        self._to_fix_act_ordering_dict = None
        self._to_fix_qkv_ordering_dict = {
            "modules": [
                "self_attention.linear_qkv",
            ],
            "layer_re": self.layer_re
        }
        super().__init__(src_names, layer_offset)

    def map_src_to_dst(self):
        for src_name in self.src_names:
             # convert word embeddings.
            if src_name in self.embedding_sync_map:
                self._dst_names.append(self.get_dst_name(self.embedding_sync_map, src_name))
                continue

            # final layer
            if src_name in self.final_layer_sync_map:
                self._dst_names.append(self.get_dst_name(self.final_layer_sync_map, src_name))
                continue

            m = self.layer_re.match(src_name)
            # Stop if that's not a layer
            if m is None:
                raise RuntimeError(f"expect src_name ({src_name}) to be a layer")

            # The index of the layer.
            layer_idx = int(m.group(1)) + self.layer_offset

            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)
            # The name of the layer.
            layer_name = f"{self.dst_prefix}.layers.{layer_idx}"

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("layer_norm") and weight_or_bias == 'weight':
                if op_name == "self_attention.linear_qkv.layer_norm":
                    ln_name = "input_layernorm"
                elif op_name == "mlp.linear_fc1.layer_norm":
                    ln_name = "post_attention_layernorm"
                else:
                    assert False, f"expect op_name ({op_name}) to be layer norm"
                self._dst_names.append(layer_name + "." + ln_name + "." + weight_or_bias)

            # Transpose the QKV matrix.
            elif op_name == "self_attention.linear_qkv" and weight_or_bias == 'weight':
                self._dst_names.append(layer_name + ".self_attn.qkv_proj.weight")

            # Transpose the weights.
            elif weight_or_bias == "weight":
                out_name = self.get_dst_name(self.layer_sync_map, op_name)
                self._dst_names.append(layer_name + out_name + "weight")

            # Ignore biases and extra_states.
            elif weight_or_bias in ["bias", "_extra_state"]:
                pass

            # Copy the rest.
            else:
                out_name = self.get_dst_name(self.layer_sync_map, op_name)
                self._dst_names.append(layer_name + out_name)

class MCore2Qwen2SyncMap(MCore2LlamaSyncMap):
    """sync map:megatron-core qwen2 to huggingface transformer"""
    def map_src_to_dst(self):
        for src_name in self.src_names:
             # convert word embeddings.
            if src_name in self.embedding_sync_map:
                self._dst_names.append(self.get_dst_name(self.embedding_sync_map, src_name))
                continue

            # final layer
            if src_name in self.final_layer_sync_map:
                self._dst_names.append(self.get_dst_name(self.final_layer_sync_map, src_name))
                continue

            m = self.layer_re.match(src_name)
            # Stop if that's not a layer
            if m is None:
                raise RuntimeError(f"expect src_name ({src_name}) to be a layer")

            # The index of the layer.
            layer_idx = int(m.group(1)) + self.layer_offset

            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)
            # The name of the layer.
            layer_name = f"{self.dst_prefix}.layers.{layer_idx}"

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("layer_norm") and weight_or_bias == 'weight':
                if op_name == "self_attention.linear_qkv.layer_norm":
                    ln_name = "input_layernorm"
                elif op_name == "mlp.linear_fc1.layer_norm":
                    ln_name = "post_attention_layernorm"
                else:
                    assert False, f"expect op_name ({op_name}) to be layer norm"
                self._dst_names.append(layer_name + "." + ln_name + "." + weight_or_bias)

            # Transpose the QKV matrix.
            elif op_name == "self_attention.linear_qkv":
                self._dst_names.append(layer_name + f".self_attn.qkv_proj.{weight_or_bias}")

            # Transpose the weights.
            elif weight_or_bias == "weight":
                out_name = self.get_dst_name(self.layer_sync_map, op_name)
                self._dst_names.append(layer_name + out_name + "weight")

            # Ignore biases and extra_states.
            elif weight_or_bias in ["bias", "_extra_state"]:
                pass

            # Copy the rest.
            else:
                out_name = self.get_dst_name(self.layer_sync_map, op_name)
                self._dst_names.append(layer_name + out_name)

class MCore2MoonlightSyncMap(ParameterSyncMap):
    """
        Mapping parameter keys of a MCore Moonlight (DSK3 w/o MTP) to huggingface.

        The MCore model should be with the following options:
            --moe-grouped-gemm
            --moe-router-enable-expert-bias
            --transformer-impl transformer_engine
            --multi-latent-attention
            --moe-layer-freq ([0]*1+[1]*26)
            # --mtp-num-layers 0
    """
    def __init__(self, src_names, layer_offset):
        src_prefix = "module.module"
        dst_prefix = "model"
        # The regex to extract layer names.
        #! add ([0-9]*) to extract experts_id for grouped GEMM
        self.layer_re = re.compile(rf"{src_prefix}.decoder.layers\.(\d+)\.([a-z0-9_.]+)[\._]([a-z]+)([0-9]*)")
        self.src_prefix = src_prefix
        self.dst_prefix = dst_prefix
        # vLLM skips loading rotary_pos_emb and re-initializes it. Thus, we don't synchronize it from MCore to vllm.
        self._embedding_sync_map = {
            f"{src_prefix}.embedding.word_embeddings.weight": f"{dst_prefix}.embed_tokens.weight",
        }
        self._layer_sync_map = {
            # NOTE: MLA attn
            'self_attention.linear_q_proj': '.self_attn.q_proj.', # only for q_lora_rank is None
            'self_attention.linear_q_down_proj': '.self_attn.q_a_proj.',
            'self_attention.linear_q_up_proj': '.self_attn.q_b_proj.',
            'self_attention.linear_kv_down_proj': '.self_attn.kv_a_proj_with_mqa.',
            'self_attention.linear_kv_up_proj': '.self_attn.kv_b_proj.',
            "self_attention.linear_proj": ".self_attn.o_proj.",
            # NOTE: attn
            "self_attention.linear_qkv": ".self_attn.qkv_proj.",

            # NOTE: MoE layer
            'mlp.router': '.mlp.gate.',
            'mlp.router.expert': '.mlp.gate.e_score_correction_',

            # NOTE: Experts (GroupedGEMM) source: linear_fc.weight{i}
            'mlp.experts.linear_fc1': ".mlp.experts.w13_weight", # w13 shape: (num_experts, 2 * intermediate_size, hidden_size)
            'mlp.experts.linear_fc2': ".mlp.experts.w2_weight", # w2 shape: (num_experts, hidden_size,, intermediate_size)

            # NOTE: Shared Experts
            'mlp.shared_experts.gate': '.mlp.shared_expert_gate.',
            'mlp.shared_experts.linear_fc1': '.mlp.shared_experts.gate_up_proj.',
            'mlp.shared_experts.linear_fc2': '.mlp.shared_experts.down_proj.',

            # NOTE: Dense MLP
            "mlp.linear_fc1": ".mlp.gate_up_proj.",
            "mlp.linear_fc2": ".mlp.down_proj.",
        }

        self._final_layer_sync_map = {
            f"{src_prefix}.decoder.final_layernorm.weight": f"{dst_prefix}.norm.weight",
            f"{src_prefix}.output_layer.weight": "lm_head.weight"
        }
        self._concat_params_dict = None
        self._to_fix_shared_expert_ordering = None
        self._to_allgather_routed_experts_dict = None
        self._to_alltoall_routed_experts_dict = None
        self._to_fix_act_ordering_dict = None
        self._to_fix_qkv_ordering_dict = {
            "modules": [
                "self_attention.linear_qkv",
            ],
            "layer_re": self.layer_re
        }
        super().__init__(src_names, layer_offset)

    def map_src_to_dst(self):
        # NOTE: we only mapping the first linear_fc.weight{i} to dst
        is_expert_set = set()
        for src_name in self.src_names:
             # convert word embeddings.
            if src_name in self.embedding_sync_map:
                self._dst_names.append(self.get_dst_name(self.embedding_sync_map, src_name))
                continue

            # final layer
            if src_name in self.final_layer_sync_map:
                self._dst_names.append(self.get_dst_name(self.final_layer_sync_map, src_name))
                continue

            m = self.layer_re.match(src_name)
            # Stop if that's not a layer
            if m is None:
                raise RuntimeError(f"expect src_name ({src_name}) to be a layer")

            # The index of the layer.
            #! NOTE: if vllm applies PP > 1, the layer_idx may not be GLOBAL LAYER ID
            layer_idx = int(m.group(1)) + self.layer_offset
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)
            # The name of the layer.
            layer_name = f"{self.dst_prefix}.layers.{layer_idx}"

            # For layernorm(s), simply store the layer norm.
            if (op_name.endswith("layer_norm") or op_name.endswith("layernorm")) and weight_or_bias == 'weight':
                if op_name == "self_attention.linear_q_up_proj.layer_norm": # qk_layernorm for MoE Layer
                    ln_name = "self_attn.q_a_layernorm"
                elif op_name == "self_attention.linear_kv_up_proj.layer_norm": # qk_layernorm for MoE Layer
                    ln_name = "self_attn.kv_a_layernorm"
                elif op_name == 'self_attention.q_layernorm': # qk_layernorm for Dense Layer
                    ln_name = "self_attn.q_norm"
                elif op_name == 'self_attention.k_layernorm': # qk_layernorm for Dense Layer
                    ln_name = "self_attn.k_norm"
                elif op_name == 'self_attention.linear_qkv.layer_norm':
                    ln_name = "input_layernorm" # input_layernorm for Dense Layer
                elif op_name == "input_layernorm":
                    ln_name = "input_layernorm" # input_layernorm for MoE Layer
                elif op_name == 'pre_mlp_layernorm':
                    ln_name = "post_attention_layernorm" # MoE layer
                elif op_name == "mlp.linear_fc1.layer_norm":
                    ln_name = "post_attention_layernorm" # Dense Layer
                else:
                    raise ValueError(f"expect op_name ({op_name}) to be layer norm")
                self._dst_names.append(layer_name + "." + ln_name + "." + weight_or_bias)

            # Transpose the QKV matrix.
            elif op_name == "self_attention.linear_qkv":
                self._dst_names.append(layer_name + f".self_attn.qkv_proj.{weight_or_bias}")

            elif 'mlp.experts' in op_name:
                out_name = layer_name + self.get_dst_name(self.layer_sync_map, op_name)
                if out_name not in is_expert_set:
                    self._dst_names.append(out_name)
                    is_expert_set.add(out_name)

            # NOTE: Why 'transpose'?
            # Transpose the weights.
            elif (
                (op_name == 'mlp.router.expert' and weight_or_bias == 'bias') or
                weight_or_bias == "weight"
            ):
                out_name = self.get_dst_name(self.layer_sync_map, op_name)
                self._dst_names.append(layer_name + out_name + weight_or_bias)

            # Ignore biases and extra_states.
            elif weight_or_bias in ["bias", "_extra_state"]:
                # moonlight is w/ --disable-bias-linear and w/o --add-qkv-bias
                pass
            # Copy the rest.
            else:
                out_name = self.get_dst_name(self.layer_sync_map, op_name)
                self._dst_names.append(layer_name + out_name)

def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='vLLM Arguments',
                                     allow_abbrev=False)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Args from environment
    args.rank = int(os.getenv('RANK', '0'))
    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    return args


def _init_distributed_environment(args):
    """Initialize the distributed environment."""
    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            torch.cuda.set_device(device)

    if torch.distributed.is_initialized():
        world_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match args.world_size "
                f"({torch_world_size} vs. {args.world_size}).")
    else:
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            timeout=timedelta(minutes=args.distributed_timeout_minutes))

    if CURRENT_VLLM_VERSION == VLLMVersion.v_0_8_5:
        _WORLD = None
        if _WORLD is None:
            ranks = list(range(torch.distributed.get_world_size()))
            _WORLD = init_world_group(ranks, args.local_rank, args.distributed_backend)
        else:
            assert _WORLD.world_size == torch.distributed.get_world_size(), (
                "world group already initialized with a different world size")
        mpu._WORLD = _WORLD

    initialize_model_parallel(args.tensor_model_parallel_size,
                              args.pipeline_model_parallel_size)


def initialize_vllm( # pylint: disable=dangerous-default-value,useless-return
    extra_args_provider=None,
    ignore_unknown_args=False,
    allow_no_cuda=False,
    args_dict=None
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."
    # Parse arguments
    args = parse_args(extra_args_provider, ignore_unknown_args)
    if args_dict:
        for key, value in args_dict.items():
            if value == "None":
                value = None
            if hasattr(args, key):
                default_value = getattr(args, key)
                if default_value is not None and value is not None:
                    default_type = type(default_value)
                    if not isinstance(value, default_type):
                        value = default_type(value)
            setattr(args, key, value)
    def finish_mpu_init():
        # Pytorch distributed.
        _init_distributed_environment(args)

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))


    finish_mpu_init()

    return args

def split_attn_state(param, num_heads, num_query_groups, kv_channels, hidden_size):
    nh = num_heads
    ng = num_query_groups
    dim = kv_channels
    if len(param.shape) == 1:
        param = param.view((ng, dim*nh//ng+dim*2, 1))
        q_proj = param[:, :dim*nh//ng, :].reshape(-1).contiguous()
        k_proj = param[:, dim*nh//ng:dim*nh//ng+dim, :].reshape(-1).contiguous()
        v_proj = param[:, dim*nh//ng+dim:, :].reshape(-1).contiguous()
    else:
        param = param.view((ng, dim*nh//ng+dim*2, hidden_size))
        q_proj = param[:, :dim*nh//ng, :].reshape(-1, hidden_size).contiguous()
        k_proj = param[:, dim*nh//ng:dim*nh//ng+dim, :].reshape(-1, hidden_size).contiguous()
        v_proj = param[:, dim*nh//ng+dim:, :].reshape(-1, hidden_size).contiguous()
    return torch.concat([q_proj, k_proj, v_proj], dim=0)


def fix_qwen_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param

def vllm_use_v1():
    if "VLLM_USE_V1" not in os.environ:
        os.environ["VLLM_USE_V1"] = "1"
    return bool(int(os.getenv("VLLM_USE_V1", "1")))
