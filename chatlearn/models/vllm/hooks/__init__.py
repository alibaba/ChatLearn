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
"""vLLM Hooks."""

import importlib
import os
import warnings


if importlib.util.find_spec("vllm"):

    from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion

    if CURRENT_VLLM_VERSION == VLLMVersion.v_0_8_5:
        from .vllm_0_8_5 import *
    else:
        warnings.warn(f"vLLM version expected in {list(member.value for member in VLLMVersion)}, while {CURRENT_VLLM_VERSION}. \
            if you want to use previous vllm version, please git checkout 4ad5912306df5d4a814dc2dd5567fcb26f5d473b")
