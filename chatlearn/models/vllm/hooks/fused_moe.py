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
"""Hooks of vllm-0.6.3 convert device_name to string."""


import inspect
from typing import Optional

# pylint: disable=unused-import,unused-argument
from vllm.model_executor.layers import fused_moe
from chatlearn.utils.utils import detect_and_insert_code_to_func


def get_config_file_name(source_code):
    pattern = 'device_name = current_platform.get_device_name().replace(" ", "_")'
    new_code = \
"""
device_name = str(current_platform.get_device_name()).replace(" ", "_")
"""
    source_code = detect_and_insert_code_to_func(source_code, pattern, new_code, line_offset=1, replace=True)
    return source_code

source = inspect.getsource(fused_moe.get_config_file_name)
if 'current_platform.get_device_name' in source:
    exec(get_config_file_name(source)) # pylint: disable=exec-used
    fused_moe.get_config_file_name = get_config_file_name

    
