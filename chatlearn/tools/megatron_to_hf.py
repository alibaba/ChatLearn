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
"""Convert Megatron checkpoint to Huggingface Transformers checkpoint."""
import argparse
import json
import glob
import os
import re
import sys
import torch
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint


def add_checkpointing_args(parser):
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument(
        "--vocab_dir",
        type=str,
        help="Vocab dir.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['llama'],
        default="llama",
        help="model type.",
    )
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )
    parser.add_argument(
        "--megatron_path",
        type=str,
        default=None,
        help=(
            "Path to Megatron-LM"
        ),
    )
    parser.add_argument(
        "--use_legacy_models",
        action="store_true",
        help=(
            "Whether using legacy models. Default: False."
        )
    )
    parser.add_argument(
        "--validate_checkpoint",
        action="store_true",
        help=(
            "Whether validating converted checkpoint. Default: False."
        )
    )
    return parser


# The simple map of names for "automated" rules.
megatron_to_transformers = {
    "self_attention.dense": ".self_attn.o_proj.",
    "mlp.dense_h_to_4h_1": ".mlp.gate_proj.",
    "mlp.dense_h_to_4h_2": ".mlp.up_proj.",
    "mlp.dense_4h_to_h": ".mlp.down_proj.",
    "self_attention.rotary_emb":".self_attn.rotary_emb.inv_freq"
}


tensor_parallel_params_mg = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight"
]


mcore_to_transformers = {
    "self_attention.linear_proj":".self_attn.o_proj.",
    "linear_fc1_1":".w1.",
    "linear_fc1_2":".w3.",
    "linear_fc2":".w2.",
    "mlp.router":".block_sparse_moe.gate.",
    "self_attention.rotary_emb":".self_attn.rotary_emb.inv_freq" # unneeded for MoE
}

def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def split_attn_state(param, args):
    nh = args.num_attention_heads
    ng = args.num_query_groups if args.group_query_attention else args.num_attention_heads
    dim = args.kv_channels
    param = param.view((ng, dim*nh//ng+dim*2, args.hidden_size))
    q_proj = param[:, :dim*nh//ng, :].reshape(-1, args.hidden_size).contiguous()
    k_proj = param[:, dim*nh//ng:dim*nh//ng+dim, :].reshape(-1, args.hidden_size).contiguous()
    v_proj = param[:, dim*nh//ng+dim:, :].reshape(-1, args.hidden_size).contiguous()
    return q_proj, k_proj, v_proj


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
        checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


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


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Load Megatron-LM checkpoint arguments from the state dict
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]

    for root, dirnames, _ in os.walk(args.load_path):
        for dirname in dirnames:
            if dirname in possible_sub_dirs:
                rank0_checkpoint_name = glob.glob(os.path.join(root, dirname) + "/*.pt")
                args.load_path = root
                rank0_checkpoint_path = rank0_checkpoint_name[0]

    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    output_state_dict = {}

    checkpoint_version = state_dict.get("checkpoint_version", 0.0)
    assert checkpoint_version >= 3.0
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    assert tp_size == 1 and pp_size == 1
    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Convert.
    print("Converting")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)

    # Convert and store the position embeddings.
    position_embeddings = get_element_from_dict_by_path(
        tp_state_dicts[0], "model.language_model.embedding.position_embeddings.weight"
    )

    if position_embeddings:
        output_state_dict["transformer.position_embeddings.weight"] = position_embeddings.to(dtype)

    # Convert and store the word embeddings.
    word_embeddings = []

    for tp_rank in range(tp_size):
        embeddings = get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.word_embeddings_for_head.weight"
            )
        # After training with megatron, word_embeddings is stored differently
        if isinstance(embeddings, dict):
            embeddings = get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"
            )
        word_embeddings.append(embeddings)

    word_embeddings = torch.cat(word_embeddings, dim=0)
    word_embeddings = word_embeddings.to(dtype)
    output_state_dict["model.embed_tokens.weight"] = word_embeddings

    # Transformer Layers
    print("Converting transformer layers")
    num_layers = megatron_args.num_layers // pp_size

    for pp_rank in range(pp_size):
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
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            # Match the name.
            m = layer_re.match(key)
            # Stop if that's not a layer
            if m is None:
                break
            if val is None:
                continue

            # The index of the layer.
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)
            # The name of the layer.
            layer_name = f"model.layers.{layer_idx}"

            if op_name + "." + weight_or_bias not in tensor_parallel_params_mg:
                params = val.to(dtype)
            else:
                dim = 1 if op_name in ["self_attention.dense", "mlp.dense_4h_to_h"] else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            # For norm(s), simply store the norm.
            if op_name.endswith("norm") and weight_or_bias == 'weight':
                ln_name = "input_layernorm" if op_name.startswith("input") else "post_attention_layernorm"
                output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = params

            # Transpose the QKV matrix.
            elif op_name in ("attention.query_key_value", "self_attention.query_key_value") \
                and weight_or_bias == "weight":
                q_proj, k_proj, v_proj = split_attn_state(params, megatron_args)
                if args.model_type == "llama":
                    output_state_dict[layer_name + ".self_attn.q_proj.weight"] = q_proj
                    output_state_dict[layer_name + ".self_attn.k_proj.weight"] = k_proj
                    output_state_dict[layer_name + ".self_attn.v_proj.weight"] = v_proj
            # Transpose the weights.
            elif weight_or_bias == "weight":
                if 'h_to_4h' in op_name:
                    h_to_4h_1, h_to_4h_2 = torch.split(params, params.size(0)//2, 0)
                    out_name = megatron_to_transformers[op_name+'_1']
                    output_state_dict[layer_name + out_name + "weight"] = h_to_4h_1
                    out_name = megatron_to_transformers[op_name+'_2']
                    output_state_dict[layer_name + out_name + "weight"] = h_to_4h_2
                else:
                    out_name = megatron_to_transformers[op_name]
                    output_state_dict[layer_name + out_name + "weight"] = params
            # Ignore them
            elif weight_or_bias == "bias":
                pass
            # Copy the Rotary Embedding
            else:
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name] = params

    if megatron_args.num_layers != (layer_idx + 1):
        raise ValueError(f"Expected {megatron_args.num_layers} layers but found {layer_idx + 1}")

    # The final norm.
    print("Converting final norm")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    output_state_dict["model.norm.weight"] = params["final_norm.weight"].to(dtype)

    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    params = torch.cat([
                        get_element_from_dict_by_path(tp_state_dicts[i], 'model.language_model.output_layer.weight')
                        for i in range(tp_size)]
        )
    output_state_dict["lm_head.weight"] = params.to(dtype)

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Store the state_dict to file.
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # Save the model
    if not os.path.exists(args.save_path):
        os.system(f'mkdir -p {args.save_path}')
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )

    # Saving config and tokenzier files
    for fn in glob.glob(args.vocab_dir + "/*"):
        if (fn.endswith(".json") or fn.endswith("tokenizer.model") or fn.endswith(".py")) and 'pytorch_model.bin.index.json' not in fn:
            os.system(f"cp {fn} {args.save_path}")


def convert_checkpoint_from_mcore_to_transformers(args):
    """
    Convert NVIDIA MCore checkpoint to HuggingFace Transformers checkpoint. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality.

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Load Megatron-Core checkpoint arguments from the state dict
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000", "mp_rank_00_000_000"]
    for root, dirnames, _ in os.walk(args.load_path):
        for dirname in dirnames:
            if dirname in possible_sub_dirs:
                rank0_checkpoint_name = glob.glob(os.path.join(root, dirname) + "/*.pt")
                args.load_path = root
                rank0_checkpoint_path = rank0_checkpoint_name[0]

    print(f"Loading Megatron-Core checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-Core checkpoint does not contain arguments. This utility only supports Megatron-Core checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-Core checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    output_state_dict = {}

    checkpoint_version = state_dict.get("checkpoint_version", 0.0)
    assert checkpoint_version >= 3.0
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    ep_size = megatron_args.expert_model_parallel_size
    assert tp_size == 1 and pp_size == 1 and ep_size == 1

    # Possible keys for MoE models:
    # 'embedding.word_embeddings.weight',
    # 'decoder.layers.0.self_attention.linear_proj.weight',
    # 'decoder.layers.0.self_attention.linear_proj._extra_state',
    # 'decoder.layers.0.self_attention.linear_qkv.layer_norm_weight',
    # 'decoder.layers.0.self_attention.linear_qkv.weight',
    # 'decoder.layers.0.self_attention.linear_qkv._extra_state',
    # 'decoder.layers.0.pre_mlp_layernorm.weight',
    # 'decoder.layers.0.mlp.router.weight',
    # 'decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight',
    # 'decoder.layers.0.mlp.experts.local_experts.0.linear_fc1._extra_state',
    # 'decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight',
    # ...,
    # 'decoder.final_layernorm.weight',
    # 'output_layer.weight',
    # 'output_layer._extra_state',
    # 'decoder'
    # The regex to extract layer names.
    layer_re = re.compile(r"decoder.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z_]+)")
    expert_re = re.compile(r"decoder.layers\.(\d+)\.([a-z0-9_.]+)\.(\d+)\.([a-z0-9_.]+)\.(weight|bias|_extra_state)")

    # Convert.
    print("Converting")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)
    state_dict = tp_state_dicts[0]['model']

    # Convert and store the position embeddings.
    position_embeddings = state_dict.get("embedding.position_embeddings.weight", None)
    if position_embeddings:
        output_state_dict["transformer.position_embeddings.weight"] = position_embeddings.to(dtype)

    # Convert and store the word embeddings.
    word_embedding = state_dict.get("embedding.word_embeddings.weight", None)
    output_state_dict["model.embed_tokens.weight"] = word_embedding.to(dtype)

    # Transformer Layers
    print("Converting transformer layers")

    def process_dense(layer_match_res, output_state_dict):
        # The name of the operation.
        op_name = layer_match_res.group(2)
        # Is it a weight or a bias?
        weight_or_bias = layer_match_res.group(3)

        # Ignore them
        if weight_or_bias in ('bias', '_extra_state'):
            return
        params = val.to(dtype)

        # For norm(s), simply store the norm.
        if weight_or_bias.endswith("norm_weight"): # e.g. self_attention.linear_qkv.layer_norm_weight
            ln_name = "input_layernorm"
            output_state_dict[layer_name + "." + ln_name + ".weight"] = params
        elif op_name.endswith("norm") and weight_or_bias == 'weight': # e.g. pre_mlp_layernorm.weight
            ln_name = "post_attention_layernorm"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = params

        # Transpose the QKV matrix.
        elif op_name == "self_attention.linear_qkv" \
            and weight_or_bias == "weight":
            q_proj, k_proj, v_proj = split_attn_state(params, megatron_args)
            if args.model_type == "llama":
                output_state_dict[layer_name + ".self_attn.q_proj.weight"] = q_proj
                output_state_dict[layer_name + ".self_attn.k_proj.weight"] = k_proj
                output_state_dict[layer_name + ".self_attn.v_proj.weight"] = v_proj

        # Store other weights such as router
        elif weight_or_bias == "weight":
            out_name = mcore_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "weight"] = params

        # Copy the Rotary Embedding
        else:
            out_name = mcore_to_transformers[op_name]
            output_state_dict[layer_name + out_name] = params

    def process_moe(expert_match_res, output_state_dict):
        # The prefix of the expert
        expert_prefix = expert_match_res.group(2)
        # The index of the expert
        expert_idx = expert_match_res.group(3)
        # the name of the operation
        op_name = expert_match_res.group(4)
        # Is it a weight or a bias?
        weight_or_bias = expert_match_res.group(5)

        # Ignore them
        if weight_or_bias in ('bias', '_extra_state'):
            return
        params = val.to(dtype)

        expert_name = f".block_sparse_moe.experts.{expert_idx}"
        if 'linear_fc1' in op_name:
            linear_fc1_1, linear_fc1_2 = torch.split(params, params.size(0)//2, 0)
            out_name = mcore_to_transformers[op_name+'_1']
            output_state_dict[layer_name + expert_name + out_name + "weight"] = linear_fc1_1
            out_name = mcore_to_transformers[op_name+'_2']
            output_state_dict[layer_name + expert_name + out_name + "weight"] = linear_fc1_2
        elif 'linear_fc2' in op_name:
            out_name = mcore_to_transformers[op_name]
            output_state_dict[layer_name + expert_name + out_name + "weight"] = params
        else:
            assert False, f"Unrecognized MoE module {expert_prefix}.{expert_idx}.{op_name}"

    # Extract the layers.
    for key, val in state_dict.items():
        # Match the name.
        layer_match_res = layer_re.match(key)
        expert_match_res = expert_re.match(key)
        # Continue if that's not a layer
        if layer_match_res is None:
            continue
        if val is None:
            continue

        # The index of the layer.
        layer_idx = int(layer_match_res.group(1))
        # The name of the layer.
        layer_name = f"model.layers.{layer_idx}"

        if expert_match_res: # Deal with sparse layers
            process_moe(expert_match_res, output_state_dict)
        else: # Deal with dense layers
            process_dense(layer_match_res, output_state_dict)

    if megatron_args.num_layers != (layer_idx + 1):
        raise ValueError(f"Expected {megatron_args.num_layers} layers but found {layer_idx + 1}")

    # The final norm.
    print("Converting final norm")
    params = state_dict.get("decoder.final_layernorm.weight", None)
    output_state_dict["model.norm.weight"] = params.to(dtype)

    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    params = state_dict.get('output_layer.weight', None)
    output_state_dict["lm_head.weight"] = params.to(dtype)

    print("Saving checkpoint...")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Store the state_dict to file.
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # Save the model
    if not os.path.exists(args.save_path):
        os.system(f'mkdir -p {args.save_path}')
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )

    # Saving config and tokenzier files
    for fn in glob.glob(args.vocab_dir + "/*"):
        if (fn.endswith(".json") or fn.endswith("tokenizer.model") or fn.endswith(".py")) and not fn.endswith(".index.json"):
            os.system(f"cp {fn} {args.save_path}")

    # It should be done!
    print("Conversion from Megatron-Core to Transformers is done!")

# pylint: disable=import-outside-toplevel
def validate_loading_checkpoints(args):
    from transformers import AutoModelForCausalLM
    _, model_loading_info = AutoModelForCausalLM.from_pretrained(args.save_path, output_loading_info=True)
    if len(model_loading_info["missing_keys"]) > 0:
        assert False, f"Invalid model checkpoint on missing_keys: {model_loading_info['missing_keys']}"
    if len(model_loading_info["unexpected_keys"]) > 0:
        assert False, f"Invalid model checkpoint on unexpected_keys: {model_loading_info['unexpected_keys']}"
    if len(model_loading_info["mismatched_keys"]) > 0:
        assert False, f"Invalid model checkpoint on mismatched_keys: {model_loading_info['mismatched_keys']}"
    if len(model_loading_info["error_msgs"]) > 0:
        assert False, f"Invalid model checkpoint on error_msgs: {model_loading_info['error_msgs']}"

def main():
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    args = parser.parse_args()
    if args.megatron_path:
        sys.path.append(args.megatron_path)

    if args.use_legacy_models:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        convert_checkpoint_from_mcore_to_transformers(args)

    if args.validate_checkpoint:
        print("Validating converted checkpoints...")
        validate_loading_checkpoints(args)
        print("Validation success!")


if __name__ == "__main__":
    main()
