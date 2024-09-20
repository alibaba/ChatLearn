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
"""test checkpoint conversion between megatron (legacy or mcore) and huggingface"""

import argparse

import torch
from transformers import AutoModel

def extract_name_and_params(model):
    name_list = []
    param_list = []
    model_named_parameters = model.named_parameters()
    for name, param in model_named_parameters:
        name_list.append(name)
        param_list.append(param)
    return name_list, param_list

def compare_checkpoint(src_path, dst_path):
    src_model = AutoModel.from_pretrained(src_path)
    dst_model = AutoModel.from_pretrained(dst_path)
    src_model_names, src_model_params = extract_name_and_params(src_model)
    dst_model_names, dst_model_params = extract_name_and_params(dst_model)
    assert src_model_names == dst_model_names
    for i, (src_param, dst_param) in enumerate(zip(src_model_params, dst_model_params)):
        print(f"Comparing {src_model_names[i]}")
        assert torch.equal(src_param, dst_param), f"Parameter {src_model_names[i]} is not equal for two models."
    return True

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src-path', type=str, required=True,
                        help='source huggingface checkpoint path')
    parser.add_argument('--dst-path', type=str, required=True,
                        help='destinate hugginface checkpoint path')
    args = parser.parse_args()

    return compare_checkpoint(args.src_path, args.dst_path)

if __name__ == '__main__':
    main()
