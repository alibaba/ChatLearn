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
import glob
import operator
import os
import re
import random
import subprocess
import sys

from datetime import timedelta
from functools import reduce
import numpy as np

import torch
import torch.distributed

from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion

try:
    from chatlearn.utils.megatron_import_helper import update_num_microbatches
    from chatlearn.utils.megatron_import_helper import find_checkpoint_rank_0
    from chatlearn.utils.megatron_import_helper import fix_query_key_value_ordering
    from chatlearn.utils.megatron_import_helper import get_checkpoint_tracker_filename
    from chatlearn.utils.megatron_import_helper import get_checkpoint_version
    from chatlearn.utils.megatron_import_helper import set_checkpoint_version
    from chatlearn.utils.megatron_import_helper import read_metadata
    from chatlearn.utils.megatron_import_helper import unwrap_model
except ImportError:
    print("Cannot import megatron, please set megatron python path first.")

try:
    from chatlearn.utils.vllm_import_helper import init_world_group
except ImportError:
    print("Cannot import init_world_group for vLLM 0.5.1, please install vLLM 0.5.1 first.")

try:
    from chatlearn.utils.vllm_import_helper import get_pipeline_model_parallel_rank
    from chatlearn.utils.vllm_import_helper import get_pipeline_model_parallel_world_size
    from chatlearn.utils.vllm_import_helper import _set_default_torch_dtype
    from chatlearn.utils.vllm_import_helper import parallel_state as mpu
    from chatlearn.utils.vllm_import_helper import initialize_model_parallel
    from chatlearn.utils.vllm_import_helper import initialize_dummy_weights
except ImportError:
    print("Cannot import vllm, please install vllm 0.3.0 or 0.5.1 first.")

from .constant import QwenVersion


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
        "mlp.dense_layernorm": "mlp.dense_layernorm",
}


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


class Megatron2LlamaSyncMap(ParameterSyncMap):
    """sync map:megatron to llama transformer"""
    def __init__(self, src_names, layer_offset):
        src_prefix = "module.module.language_model"
        dst_prefix = "model.model"
        # The regex to extract layer names.
        self.layer_re = re.compile(f"{src_prefix}.encoder.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)") # pylint: disable=anomalous-backslash-in-string
        self.src_prefix = src_prefix
        self.dst_prefix = dst_prefix
        self._embedding_sync_map = {
            f"{src_prefix}.embedding.word_embeddings.weight": f"{dst_prefix}.embed_tokens.weight"
        }
        self._layer_sync_map = {
            "self_attention.dense": ".self_attn.o_proj.",
            "mlp.dense_h_to_4h": ".mlp.gate_up_proj.",
            "mlp.dense_4h_to_h": ".mlp.down_proj.",
            "self_attention.rotary_emb":".self_attn.rotary_emb.inv_freq",
        }
        self._final_layer_sync_map = {
            f"{src_prefix}.encoder.final_norm.weight": f"{dst_prefix}.norm.weight",
            f"{src_prefix}.output_layer.weight": "model.lm_head.weight"
        }
        self._concat_params_dict = None
        self._to_fix_act_ordering_dict = None
        self._to_fix_qkv_ordering_dict = {
            "modules": [
                "attention.query_key_value",
                "self_attention.query_key_value"
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
                raise RuntimeError(f"expect src_name to be a layer, while {src_name}")

            # The index of the layer.
            layer_idx = int(m.group(1)) + self.layer_offset

            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)
            # The name of the layer.
            layer_name = f"{self.dst_prefix}.layers.{layer_idx}"

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("_norm") and weight_or_bias == 'weight':
                ln_name = "input_layernorm" if op_name.startswith("input") else "post_attention_layernorm"
                self._dst_names.append(layer_name + "." + ln_name + "." + weight_or_bias)

            # Transpose the QKV matrix.
            elif op_name in ["attention.query_key_value", "self_attention.query_key_value"] and  \
                    weight_or_bias == "weight":
                self._dst_names.append(layer_name + ".self_attn.qkv_proj.weight")

            # Transpose the weights.
            elif weight_or_bias == "weight":
                out_name = self.get_dst_name(self.layer_sync_map, op_name)
                self._dst_names.append(layer_name + out_name + "weight")

            # Copy the bias.
            # Ignore them
            elif weight_or_bias == "bias":
                pass

            # Copy the Rotary Embedding
            else:
                out_name = self.get_dst_name(self.layer_sync_map, op_name)
                self._dst_names.append(layer_name + out_name)


class Megatron2QWenSyncMap(ParameterSyncMap):
    """sync map:megatron to qwen transformer"""
    def __init__(self, src_names, layer_offset, qwen_version=QwenVersion.v_1.value):
        self.qwen_version = qwen_version
        src_prefix = "module.module.language_model"

        # configuration for different versions of qwen
        if qwen_version == QwenVersion.v_1.value:
            dst_prefix = "model.transformer"
            embed_name = "wte"
            att_dense_name = ".attn.c_proj."
            self.layer_prefix = "h"
            mlp_dense_name = ".mlp.c_proj."
            final_norm = "ln_f"
        elif qwen_version == QwenVersion.v_2.value:
            dst_prefix = "model.model"
            embed_name = "embed_tokens"
            att_dense_name = ".self_attn.o_proj."
            self.layer_prefix = "layers"
            mlp_dense_name = ".mlp.down_proj."
            final_norm = "norm"
        else:
            raise RuntimeError(f"Unsupported qwen version {qwen_version}, only 1.0 or 2.0 for now.")

        # The regex to extract layer names.
        self.layer_re = re.compile(f"{src_prefix}.encoder.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)") # pylint: disable=anomalous-backslash-in-string
        self.src_prefix = src_prefix
        self.dst_prefix = dst_prefix
        self._embedding_sync_map = {
            f"{src_prefix}.embedding.word_embeddings.weight": f"{dst_prefix}.{embed_name}.weight"
        }
        self._layer_sync_map = {
            "attention.attention_layernorm": ".attn.attention_layernorm.",
            "attention.dense": ".attn.c_proj.",
            "self_attention.dense": att_dense_name,
            "mlp.dense_h_to_4h": ".mlp.gate_up_proj.",
            "mlp.w1": ".mlp.gate_up_proj.",
            "mlp.w2": ".mlp.gate_up_proj.",
            "mlp.dense_4h_to_h": mlp_dense_name,
            "mlp.dense_layernorm": "mlp.dense_layernorm",
        }
        self._final_layer_sync_map = {
            f"{src_prefix}.encoder.final_layernorm.bias": f"{dst_prefix}.{final_norm}.bias",
            f"{src_prefix}.encoder.final_layernorm.weight": f"{dst_prefix}.{final_norm}.weight",
            f"{src_prefix}.output_layer.weight": "model.lm_head.weight"
        }
        self._concat_params_dict = {
            "modules": ["mlp.w1", "mlp.w2"],
            "dim": 0
        }
        self._to_fix_act_ordering_dict = {
            "modules": ["mlp.dense_h_to_4h"],
            "dim": 0
        }
        self._to_fix_qkv_ordering_dict = {
            "modules": [
                "attention.query_key_value",
                "self_attention.query_key_value"
            ],
            "layer_re": self.layer_re
        }

        src_names_list = []
        for idx, s_name in enumerate(src_names):
            if "mlp.w1" in s_name:
                src_names_list.append(src_names[idx + 1])
                src_names_list.append(s_name)
            elif "mlp.w2" in s_name:
                continue
            else:
                src_names_list.append(s_name)
        super().__init__(src_names_list, layer_offset)

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
                raise RuntimeError(f"expect src_name to be a layer, while {src_name}")
            # The index of the layer.
            layer_idx = int(m.group(1)) + self.layer_offset

            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)
            # The name of the layer.
            layer_name = f"{self.dst_prefix}.{self.layer_prefix}.{layer_idx}"

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("layernorm"):

                if self.qwen_version == QwenVersion.v_1.value:
                    if "attention." in op_name:
                        self._dst_names.append(
                            layer_name + self.get_dst_name(self.layer_sync_map, ".attn.attention_layernorm.") + weight_or_bias)
                    if "mlp." in op_name:
                        self._dst_names.append(
                            layer_name + self.get_dst_name(self.layer_sync_map, op_name) + weight_or_bias)
                if op_name.startswith("input"):
                    ln_name = "ln_1" if self.qwen_version == QwenVersion.v_1.value else "input_layernorm"
                    self._dst_names.append(
                        layer_name + "." + ln_name + "." + weight_or_bias)
                elif op_name.startswith("post"):
                    ln_name = "ln_2" if self.qwen_version == QwenVersion.v_1.value else "post_attention_layernorm"
                    self._dst_names.append(
                        layer_name + "." + ln_name + "." + weight_or_bias)
                elif self.qwen_version == QwenVersion.v_2.value:
                    raise RuntimeError(f"unsupport layernorm {op_name}.")

            elif op_name == "self_attention.rotary_emb":
                self._dst_names.apepnd(layer_name + ".attn.rotary_emb.inv_freq")

            # Transpose the QKV matrix and the bias.
            elif op_name in ["attention.query_key_value", "self_attention.query_key_value"]:
                if self.qwen_version == QwenVersion.v_1.value:
                    dst_name = layer_name + f".attn.c_attn.{weight_or_bias}"
                else:
                    dst_name = layer_name + f".self_attn.qkv_proj.{weight_or_bias}"
                self._dst_names.append(dst_name)

            elif op_name in ["mlp.w1", "mlp.w2"]:
                out_name =  self.layer_sync_map[op_name]
                gate_up_proj_name = layer_name + out_name + "weight"
                if gate_up_proj_name not in self._dst_names:
                    self._dst_names.append(gate_up_proj_name)

            # Transpose the weights.
            elif weight_or_bias in ["weight", "bias"]:
                out_name = self.layer_sync_map[op_name]
                self._dst_names.append(layer_name + out_name + weight_or_bias)


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


def get_model(model_provider, args, need_load_ckpt=True):
    with _set_default_torch_dtype(args.get("params_dtype")):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        model = model_provider()
        model = model.cuda()
        if args["load"] and need_load_ckpt:
            model.load_weights()
        else:
            # For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
    return model.eval()


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

    if CURRENT_VLLM_VERSION == VLLMVersion.v_0_5_1.value:
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


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.
    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


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


def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.
    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = glob.glob(os.path.join(args["load"], sub_dir_name) + "/model*.pt")[0]
        checkpoint_path = os.path.join(args["load"], sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


def convert_llama_state_dict_from_megatron_to_vllm(args, hf_config, qwen_version=None):
    """Convert NVIDIA Megatron-LM state_dict to vLLM llama state_dict.

        Args:
            args (argparse.Namespace): the arguments to the script
    """
    assert qwen_version is None, f"Expect qwen_version is None for Llama, while {qwen_version}"
    # Load original state dict from Megatron-LM checkpoint.
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = get_pipeline_model_parallel_rank()
    assert pp_rank == 0
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]

    for root, dirnames, _ in os.walk(args["load"]):
        for dirname in dirnames:
            if dirname in possible_sub_dirs:
                rank0_checkpoint_name = glob.glob(os.path.join(root, dirname) + "/model*.pt")
                args["load"] = root
                rank0_checkpoint_path = rank0_checkpoint_name[0]

    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if "checkpoint_version" in state_dict.keys():
        checkpoint_version = state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    output_state_dict = {}

    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    assert pp_size == 1
    # The number of heads.
    heads = hf_config.num_attention_heads // tp_size
    # The hidden_size per head.
    hidden_size_per_head = hf_config.hidden_size // hf_config.num_attention_heads

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)") # pylint: disable=anomalous-backslash-in-string

    # Convert.
    print("Start to convert...")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)

    # Convert and store the position embeddings.
    position_embeddings = get_element_from_dict_by_path(
        tp_state_dicts[0], "model.language_model.embedding.position_embeddings.weight"
    )

    if position_embeddings:
        output_state_dict["transformer.position_embeddings.weight"] = position_embeddings.to(hf_config.torch_dtype)

    # Convert and store the word embeddings.
    word_embeddings = []

    word_embeddings = get_element_from_dict_by_path(
        tp_state_dicts[tp_rank], "model.word_embeddings_for_head.weight")
    # After training with megatron, word_embeddings is stored differently
    if isinstance(word_embeddings, dict):
        word_embeddings = get_element_from_dict_by_path(
            tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"
        )
    word_embeddings = word_embeddings.to(hf_config.torch_dtype)
    output_state_dict["model.model.embed_tokens.weight"] = word_embeddings
    # Reset the vocab size
    hf_config.vocab_size = word_embeddings.shape[0]

    # Transformer Layers
    print("Converting transformer layers")
    num_layers = hf_config.num_hidden_layers // pp_size

    if pp_size > 0:
        print(f"Converting pipeline parallel rank {pp_rank}")
        tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, pp_rank)

    # The transformer.
    path = (
        "model.language_model.transformer"
        if "transformer" in get_element_from_dict_by_path(tp_state_dicts[0], "model.language_model").keys()
        else "model.language_model.encoder"
    )

    # Extract the layers.
    for key, val in get_element_from_dict_by_path(tp_state_dicts[tp_rank], path).items():
        # skip None value.
        # TODO(jiangle.jl): whether to process empty value.
        if val is None:
            continue
        # Match the name.
        m = layer_re.match(key)
        # Stop if that's not a layer
        if m is None:
            break
        # The index of the layer.
        layer_idx = int(m.group(1)) + pp_rank * num_layers
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)
        # The name of the layer.
        layer_name = f"model.model.layers.{layer_idx}"

        params = val.to(hf_config.torch_dtype)

        # For layernorm(s), simply store the layer norm.
        if (op_name.endswith("_norm") or op_name.endswith("_layernorm")) and weight_or_bias == 'weight':
            ln_name = "input_layernorm" if op_name.startswith("input") else "post_attention_layernorm"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = params

        # Transpose the QKV matrix.
        elif op_name in ["attention.query_key_value", "self_attention.query_key_value"] and weight_or_bias == "weight":
            input_shape = params.size()
            shape = (heads, hidden_size_per_head, 3) + input_shape[1:]
            division = reduce(operator.mul, shape, 1)
            num_elements = params.numel()
            if num_elements != division:
                # model with gqa dont need to fix qkv ordering.
                output_state_dict[layer_name + ".self_attn.qkv_proj.weight"] = params
            else:
                out_val = fix_qwen_query_key_value_ordering(
                    params, checkpoint_version, 3, heads, hidden_size_per_head
                )
                # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
                out_val = out_val.contiguous()
                # Store.
                output_state_dict[layer_name + ".self_attn.qkv_proj.weight"] = out_val

        # Transpose the bias.
        elif op_name in ["attention.query_key_value", "self_attention.query_key_value"] and weight_or_bias == "bias":
            out_val = fix_qwen_query_key_value_ordering(
                params, checkpoint_version, 3, heads, hidden_size_per_head
            )
            # Store. No change of shape.
            output_state_dict[layer_name + ".self_attn.qkv_proj.bias"] = out_val


        # Transpose the weights.
        elif weight_or_bias == "weight":
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "weight"] = params

        # Copy the bias.
        # Ignore them
        elif weight_or_bias == "bias":
            pass

        # Copy the Rotary Embedding
        else:
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name] = params

    if hf_config.num_hidden_layers != (layer_idx + 1):
        raise ValueError(f"Expected {hf_config.num_hidden_layers} layers but found {layer_idx + 1}")

    # The final layernorm.
    print("Converting final layernorm")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    final_norm_weight =  params["final_norm.weight"] if "final_norm.weight" in params else params["final_layernorm.weight"]
    output_state_dict["model.model.norm.weight"] = final_norm_weight.to(hf_config.torch_dtype)

    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    params = get_element_from_dict_by_path(tp_state_dicts[tp_rank], 'model.language_model.output_layer.weight')
    output_state_dict["model.lm_head.weight"] = params.to(hf_config.torch_dtype)

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    return output_state_dict


def convert_qwen_state_dict_from_megatron_to_vllm(args, hf_config, qwen_version=QwenVersion.v_1.value):
    # The converted output model.
    output_state_dict = {}

    # configuration for different versions of qwen
    if qwen_version == QwenVersion.v_1.value:
        prefix_name = "model.transformer."
        embed_name = "wte"
        layer_prefix = "h"
        final_norm = "ln_f"
        func_map = megatron_qwen_to_transformers
    elif qwen_version == QwenVersion.v_2.value:
        prefix_name = "model.model."
        embed_name = "embed_tokens"
        layer_prefix = "layers"
        final_norm = "norm"
        func_map = megatron_qwen2_to_transformers
    else:
        raise RuntimeError(f"Unsupported qwen version {qwen_version}, only 1.0 or 2.0 for now. while {qwen_version}.")

    # Load original state dict from Megatron-LM checkpoint.
    tp_rank = mpu.get_tensor_model_parallel_rank()
    possible_sub_dirs = [f"mp_rank_{tp_rank:02d}"]

    for root, dirnames, _ in os.walk(args["load"]):
        for dirname in dirnames:
            if dirname in possible_sub_dirs:
                rank0_checkpoint_name = glob.glob(os.path.join(root, dirname) + "/*.pt")
                args["load"] = root
                rank0_checkpoint_path = rank0_checkpoint_name[0]

    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    input_state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")

    # old versions did not store training args
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        # do not make the user write a config file when the exact dimensions/sizes are already in the checkpoint
        hf_config.vocab_size = ds_args.padded_vocab_size
        hf_config.max_position_embeddings = ds_args.max_position_embeddings
        hf_config.hidden_size = ds_args.hidden_size
        hf_config.num_hidden_layers = ds_args.num_layers
        hf_config.num_attention_heads = ds_args.num_attention_heads
        hf_config.intermediate_size = ds_args.ffn_hidden_size

    tp_size = ds_args.tensor_model_parallel_size
    pp_size = ds_args.pipeline_model_parallel_size
    assert pp_size == 1
    # The number of heads.
    heads = hf_config.num_attention_heads // tp_size
    # The hidden_size per head.
    hidden_size_per_head = hf_config.hidden_size // hf_config.num_attention_heads
    # Megatron-LM checkpoint version
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0
    # The model.
    model = input_state_dict["model"]
    # The language model.
    lm = model["language_model"]
    # The embeddings.
    embeddings = lm["embedding"]

    # The word embeddings.
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # Truncate the embedding table to vocab_size rows.
    word_embeddings = word_embeddings[: hf_config.vocab_size, :]
    output_state_dict[f"{prefix_name}{embed_name}.weight"] = word_embeddings

    # The transformer. now encoder
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]
    # The position embeddings.
    if "position_embeddings" in embeddings:
        pos_embeddings = embeddings["position_embeddings"]["weight"]
        # Read the causal mask dimension (seqlen). [max_sequence_length, hidden_size]
        n_positions = pos_embeddings.size(0)
        if n_positions != hf_config.max_position_embeddings:
            raise ValueError(
                f"pos_embeddings.max_sequence_length={n_positions} and hf_config.n_positions={hf_config.max_position_embeddings} don't match"
            )
        # Store the position embeddings.
        output_state_dict[f"{prefix_name}wpe.weight"] = pos_embeddings
    else:
        n_positions = hf_config.max_position_embeddings

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)") # pylint: disable=anomalous-backslash-in-string

    # Extract the layers.
    gate_up_proj = {}
    for key, val in transformer.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            break

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # The name of the layer.
        layer_name = f"{prefix_name}{layer_prefix}.{layer_idx}"

        # For layernorm(s), simply store the layer norm.
        if op_name.endswith("layernorm"):

            if qwen_version == QwenVersion.v_1.value:
                if "attention." in op_name:
                    output_state_dict[
                        layer_name + ".attn.attention_layernorm." + weight_or_bias
                    ] = val
                if "mlp." in op_name:
                    output_state_dict[
                        layer_name + "." + op_name + "." + weight_or_bias
                    ] = val

            if op_name.startswith("input"):
                ln_name = "ln_1" if qwen_version == QwenVersion.v_1.value else "input_layernorm"
                output_state_dict[
                    layer_name + "." + ln_name + "." + weight_or_bias
                ] = val
            elif op_name.startswith("post"):
                ln_name  = "ln_2" if qwen_version == QwenVersion.v_1.value else "post_attention_layernorm"
                output_state_dict[
                    layer_name + "." + ln_name + "." + weight_or_bias
                ] = val
            elif qwen_version == QwenVersion.v_2.value:
                raise RuntimeError(f"unsupport layernorm {op_name}.")

        elif op_name == "self_attention.rotary_emb":
            output_state_dict[layer_name + ".attn.rotary_emb.inv_freq"] = val

        # Transpose the QKV matrix and the bias.
        elif op_name in ["attention.query_key_value", "self_attention.query_key_value"]:
            if qwen_version == QwenVersion.v_1.value:
                out_val = fix_qwen_query_key_value_ordering(
                    val, checkpoint_version, 3, heads, hidden_size_per_head
                )
                # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
                if len(list(out_val.shape)) > 1:
                    out_val = out_val.contiguous()
                # Store.
                output_state_dict[layer_name + f".attn.c_attn.{weight_or_bias}"] = out_val
            else:
                num_query_groups = ds_args.num_query_groups if ds_args.group_query_attention else ds_args.num_attention_heads
                val = split_attn_state(val, heads, num_query_groups // tp_size, hidden_size_per_head, hf_config.hidden_size)
                # Store. No change of shape.
                output_state_dict[layer_name + f".self_attn.qkv_proj.{weight_or_bias}"] = val

        elif op_name in ["mlp.dense_h_to_4h"]:
            offset = val.shape[0] // 2
            w1 = val[:offset,:]
            w2 = val[offset:,:]
            out_name = func_map[op_name]
            out_name = layer_name + out_name + "weight"
            output_state_dict[out_name] = torch.cat([w2, w1], dim=0)

        elif op_name in ["mlp.w1", "mlp.w2"]:
            gate_up_proj[op_name] = val

            if len(gate_up_proj) == 2:
                gate_up_proj = [gate_up_proj["mlp.w2"], gate_up_proj["mlp.w1"]]
                out_name = func_map[op_name]
                gate_up_proj_name = layer_name + out_name + "weight"
                output_state_dict[gate_up_proj_name] = torch.cat(gate_up_proj, dim=0)
                gate_up_proj = {}

        # Transpose the weights.
        elif weight_or_bias == "weight":
            out_name = func_map[op_name]
            output_state_dict[layer_name + out_name + "weight"] = val

        # Copy the bias.
        elif weight_or_bias == "bias":
            out_name = func_map[op_name]
            output_state_dict[layer_name + out_name + "bias"] = val

    # DEBUG.
    assert hf_config.num_hidden_layers == layer_idx + 1

    # The final layernorm.
    output_state_dict[f"{prefix_name}{final_norm}.weight"] = transformer[
        "final_layernorm.weight"
    ]
    if "final_layernorm.bias" in output_state_dict:
        output_state_dict[f"{prefix_name}{final_norm}.bias"] = transformer[
            "final_layernorm.bias"
        ]

    # LM head
    if ds_args.untie_embeddings_and_output_weights:
        output_layer = lm["output_layer"]["weight"]
        output_state_dict["model.lm_head.weight"] = output_layer
    else:
        output_state_dict["model.lm_head.weight"] = word_embeddings

    # It should be done!
    return output_state_dict


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def _load_base_checkpoint(load_dir, rank0=False):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """

    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return nothing
    if not os.path.isfile(tracker_filename):
        if not rank0:
            print_rank_0('WARNING: could not find the metadata file {} '.format(
                tracker_filename))
            print_rank_0('    will not load any checkpoints and will start from '
                         'random')
        return None, "", False

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
    else:
        checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
        if release:
            print_rank_0(f' loading release checkpoint from {load_dir}')
        else:
            print_rank_0(f' loading checkpoint from {load_dir} at iteration {iteration}')

    if isinstance(checkpoint_name, tuple):
        checkpoint_name = checkpoint_name[0]

    # Load the checkpoint.
    try:
        state_dict = torch.load(checkpoint_name, map_location='cpu')
    except ModuleNotFoundError:
        try:
            from megatron.fp16_deprecated import loss_scaler # pylint: disable=import-outside-toplevel,unused-import
            # For backward compatibility.
            if not rank0:
                print_rank_0(' > deserializing using the old code structure ...')
            sys.modules['fp16.loss_scaler'] = sys.modules[
                'megatron.fp16_deprecated.loss_scaler']
            sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
                'megatron.fp16_deprecated.loss_scaler']
            state_dict = torch.load(checkpoint_name, map_location='cpu')
            sys.modules.pop('fp16.loss_scaler', None)
            sys.modules.pop('megatron.fp16.loss_scaler', None)
        except ImportError:
            from megatron.legacy.fp16_deprecated import loss_scaler # pylint: disable=import-outside-toplevel,unused-import
            # For backward compatibility.
            if not rank0:
                print_rank_0(' > deserializing using the old code structure ...')
            sys.modules['fp16.loss_scaler'] = sys.modules[
                'megatron.legacy.fp16_deprecated.loss_scaler']
            sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
                'megatron.legacy.fp16_deprecated.loss_scaler']
            sys.modules['megatron.model'] = sys.modules['megatron.legacy.model']
            state_dict = torch.load(checkpoint_name, map_location='cpu')
            sys.modules.pop('fp16.loss_scaler', None)
            sys.modules.pop('megatron.fp16.loss_scaler', None)
            sys.modules.pop('megatron.model', None)
    except BaseException as e:
        print_rank_0('could not load the checkpoint')
        print_rank_0(e)
        sys.exit()

    return state_dict, checkpoint_name, release


def load_checkpoint(model, optimizer, opt_param_scheduler, load_arg='load', strict=True):
    """"Transform parallel strategy for checkpoint if needed."""
    args = model.model_args
    if args.get("adaptive_parallel_strategy_on_checkpoint"):
        load_dir = args[load_arg]
        target_tp = args.get("tensor_model_parallel_size")
        target_pp = args.get("pipeline_model_parallel_size")
        state_dict, _, _ = _load_base_checkpoint(load_dir, rank0=True)
        checkpoint_args = state_dict['args']
        checkpoint_tp = checkpoint_args.tensor_model_parallel_size
        checkpoint_pp = checkpoint_args.pipeline_model_parallel_size
        if target_tp != checkpoint_tp or target_pp != checkpoint_pp:
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tools/megatron_checkpoint_utils.py")
            save_dir = load_dir[:-1] if load_dir.endswith("/") else load_dir
            save_dir = save_dir + f"-transform-tp{target_tp}-pp{target_pp}"
            if not os.path.exists(save_dir):
                if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
                    model_type = "GPT"
                    cmd = f"python {script_path} --model-type {model_type} --load-dir {load_dir} " + \
                          f"--save-dir {save_dir} --target-tensor-parallel-size {target_tp} --target-pipeline-parallel-size {target_pp}"
                    subprocess.run(cmd, shell=True, check=True)
            torch.distributed.barrier()
            args[load_arg] = save_dir
            print_rank_0(f"Using transformed checkpoint {save_dir}")
    return vllm_load_checkpoint(model, optimizer, opt_param_scheduler, load_arg=load_arg, strict=strict)


def vllm_load_checkpoint(model, optimizer, opt_param_scheduler, load_arg='load', strict=True):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = model.model_args
    load_dir = args[load_arg]

    model = [unwrap_model(model)]

    state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0=False)

    # Checkpoint not loaded.
    if state_dict is None:

        # Conditionally exit at this point.
        if args.get("exit_on_missing_checkpoint", False):
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            torch.distributed.barrier()
            sys.exit()

        # Iteration defaults to 0.
        return 0

    # Set checkpoint version.
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    # Set iteration.
    if args.get("finetune", True) or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}, exiting'.format(
                                 checkpoint_name))
                sys.exit()

    # Check arguments.
    if 'args' in state_dict and not args.get("finetune", True):
        checkpoint_args = state_dict['args']
        args["consumed_train_samples"] = getattr(checkpoint_args,
                                              'consumed_train_samples', 0)
        update_num_microbatches(consumed_samples=args["consumed_train_samples"])
        args["consumed_valid_samples"] = getattr(checkpoint_args,
                                              'consumed_valid_samples', 0)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # Model.
    if len(model) == 1:
        model[0].load_state_dict(state_dict['model'], strict=strict)
    else:
        for i in range(len(model)): # pylint: disable=consider-using-enumerate
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

    # Fix up query/key/value matrix ordering if needed.
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.get("finetune", True) and not args["no_load_optim"]:
        try:
            # Load state dict.
            if optimizer is not None:
                optimizer.load_state_dict(state_dict['optimizer'])

            # Load distributed optimizer's custom parameter state.
            if args["use_distributed_optimizer"]:
                tracker_filename = get_checkpoint_tracker_filename(load_dir)
                iteration, release = read_metadata(tracker_filename)
                model_checkpoint_name = \
                    get_checkpoint_name(load_dir, iteration, release)
                optim_checkpoint_name = \
                    get_distributed_optimizer_checkpoint_name(
                        model_checkpoint_name)
                optimizer.load_parameter_state(optim_checkpoint_name)

            # Load scheduler.
            if opt_param_scheduler is not None:
                if 'lr_scheduler' in state_dict: # backward compatbility
                    opt_param_scheduler.load_state_dict(state_dict['lr_scheduler'])
                else:
                    opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()
    else:
        if (args.get("fp16") or args.get("bf16")) and optimizer is not None:
            optimizer.reload_model_params()

    # rng states.
    if not release and ("finetune" in args and not args["finetune"]) and ("no_load_rng" in args and not args["no_load_rng"]):
        try:
            if 'rng_state' in state_dict:
                # access rng_state for data parallel rank
                if args.get("data_parallel_random_init", False):
                    rng_state = state_dict['rng_state'][mpu.get_data_parallel_rank()]
                else:
                    rng_state = state_dict['rng_state'][0]
                random.setstate(rng_state['random_rng_state'])
                np.random.set_state(rng_state['np_rng_state'])
                torch.set_rng_state(rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                # Check for empty states array
                if not rng_state['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state['rng_tracker_states'])
            else:  # backward compatability
                random.setstate(state_dict['random_rng_state'])
                np.random.set_state(state_dict['np_rng_state'])
                torch.set_rng_state(state_dict['torch_rng_state'])
                torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
                # Check for empty states array
                if not state_dict['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load rng state from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {args["load"]} '
                 f'at iteration {iteration}')

    return iteration


def get_checkpoint_name(checkpoints_path, iteration, release=False,
                        pipeline_parallel=None,
                        tensor_rank=None, pipeline_rank=None,
                        expert_parallel=None, expert_rank=None):
    """Determine the directory name for this rank's checkpoint."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)

    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = (get_pipeline_model_parallel_world_size() > 1)
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = get_pipeline_model_parallel_rank()
    if expert_parallel is None:
        expert_parallel = False #(mpu.get_expert_model_parallel_world_size() > 1)
    if expert_rank is None:
        expert_rank = 0 #mpu.get_expert_model_parallel_rank()

    # Use both the tensor and pipeline MP rank. If using the distributed
    # optimizer, then the optimizer's path must additionally include the
    # data parallel rank.
    if not pipeline_parallel:
        common_path = os.path.join(checkpoints_path, directory,
                            f'mp_rank_{tensor_rank:02d}')
    else:
        common_path = os.path.join(checkpoints_path, directory,
                f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')

    if expert_parallel:
        common_path = common_path + f'_{expert_rank:03d}'

    model_path = os.path.join(common_path, "model_optim_rng.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(common_path, "model_rng.pt")
    return model_path
