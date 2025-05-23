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
from .. import is_vllm_v2


if importlib.util.find_spec("vllm"):

    from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion

    if CURRENT_VLLM_VERSION == VLLMVersion.v_0_3_0:
        from chatlearn.models.vllm.hooks.vllm_0_3_0 import *
    elif CURRENT_VLLM_VERSION == VLLMVersion.v_0_5_1:
        from chatlearn.models.vllm.hooks.vllm_0_5_1 import *
    elif CURRENT_VLLM_VERSION == VLLMVersion.v_0_6_3:
        from chatlearn.models.vllm.hooks.vllm_0_6_3 import *
    elif CURRENT_VLLM_VERSION == VLLMVersion.v_0_6_6:
        from .vllm_0_6_6 import *
    else:
        raise RuntimeError(
            f"vLLM version expected in {list(member.value for member in VLLMVersion)}, while {CURRENT_VLLM_VERSION}.")
