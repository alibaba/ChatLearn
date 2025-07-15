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
"""Offline FSDP checkpoint merge"""

import os
import json
import glob
import shutil
import argparse

import torch
import torch.distributed.tensor

from safetensors.torch import save_file

from tqdm import tqdm

def save_safetensor_item(safetensor_name, safetensor_data, save_dir):
    save_file(safetensor_data, os.path.join(save_dir, safetensor_name))

def split_list(lst, n):
    """Split list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def qwen3_key_mapping(param, model_config):
    num_expert = model_config['num_experts']
    local_names = [param.replace('group_mlp', f"experts.{i}") for i in range(num_expert)]
    return local_names

def check_groupgemm_param(key):
    return 'group_mlp' in key

arch_mapping = {
    'Qwen3MoeForCausalLM': (check_groupgemm_param, qwen3_key_mapping)
}

def convert_checkpoint_cpu(args_input):
    iter_ = args_input.iter
    hf_dir = args_input.hf_dir
    dist_model_dir = os.path.join(args_input.ckpt_dir, str(iter_))
    save_dir = args_input.save_dir
    os.makedirs(save_dir, exist_ok=True)
    is_groupgemm = args_input.groupgemm == 1

    safetensor_file = []
    other_file = []
    for file in os.listdir(hf_dir):
        if file.endswith(".safetensors"):
            safetensor_file.append(file)
        else:
            other_file.append(file)

    with open(os.path.join(hf_dir, "model.safetensors.index.json")) as f: # pylint: disable=unspecified-encoding
        safetensor_config = json.load(f)
    weight_map = safetensor_config['weight_map']
    with open(os.path.join(hf_dir, "config.json")) as f: # pylint: disable=unspecified-encoding
        model_config = json.load(f)
    arch = model_config['architectures'][0]

    # Check whether training using groupgemm
    group_gemm = arch in arch_mapping and is_groupgemm
    if group_gemm:
        check_param_fn, key_convert_fn = arch_mapping[arch]
        num_expert = model_config['num_experts']

    dist_model_files = glob.glob(os.path.join(dist_model_dir, "model_world_size_*.pt"))
    dist_model_state_dict = []
    for file in tqdm(dist_model_files, "Read Distributed Checkpoints"):
        dist_model_state_dict.append(torch.load(file, map_location="cpu"))
    param_list = list(dist_model_state_dict[0].keys())
    safetensor_dict = {key: {} for key in safetensor_file}

    for param in tqdm(param_list, desc="Merge Weights"):
        global_tensor = torch.cat([state_dict.pop(param).to_local() for state_dict in dist_model_state_dict], dim=0)
        if group_gemm:
            if check_param_fn(param):
                # Split param for groupgemm mlp weights
                local_names = key_convert_fn(param, model_config)
                num_expert = len(local_names)
                global_tensor = torch.chunk(global_tensor, num_expert, dim=0)
                safetensor_name = weight_map[local_names[0]]
                for i in range(num_expert):
                    safetensor_dict[safetensor_name][local_names[i]] = global_tensor[i]
            else:
                safetensor_name = weight_map[param]
                safetensor_dict[safetensor_name][param] = global_tensor
        else:
            safetensor_name = weight_map[param]
            safetensor_dict[safetensor_name][param] = global_tensor

    # Save safetensor files
    for name in tqdm(safetensor_dict, "Save Safetensor Files"):
        save_safetensor_item(name, safetensor_dict[name], save_dir)

    # Copy other files
    for file in other_file:
        if not file.startswith('.'):
            src_file = os.path.join(hf_dir, file)
            dst_file = os.path.join(save_dir, file)
            shutil.copy(src_file, dst_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline Checkpoint Converter")
    parser.add_argument("--hf_dir", type=str, required=True, help="Directory to load hf config")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory to load sharded checkpoint")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save converted hf checkpoint")
    parser.add_argument("--groupgemm", type=int, choices=[0, 1], default=0, help="Whether use groupgemm for training")
    parser.add_argument("--iter", type=int, required=True, help="which iter to convert")
    args = parser.parse_args()
    convert_checkpoint_cpu(args_input=args)
