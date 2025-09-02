# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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
"""helper to collect shape infos for vLLM model"""
from typing import Dict
from torch import nn

from .sharded_tensor_info import ShardedTensorInfo

def build_sharded_info_for_huggingface_model(model: nn.Module) -> Dict[str, ShardedTensorInfo]:
    """build sharded tensor info from huggingface transformer.
    It is strongly suggested to call this function with a meta-init
    transformer.

    Args:
        model (nn.Module): The given transformer model

    Returns:
        Dict[str, ShardedTensorInfo]: A dict maps local parameter
        name to sharded_info
    """
    infos = {}
    for name, weight in model.state_dict().items():
        infos[name] = ShardedTensorInfo(
            dtype=weight.dtype,
            global_shape=weight.shape,
            axis_fragmentations=(1, ) * weight.ndim,
            global_offset=(0, ) * weight.ndim,
        )
    return infos
