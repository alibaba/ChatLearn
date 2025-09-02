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
"""constants."""

import importlib
from enum import Enum

# Regroup
REF_LIST = "ref_list"
INDEX_TAG = "data_index"

LOG_START = "chatlearn_log"

QKV_LAYER_NAME = ["query_key_value"]


# vLLM version
CURRENT_VLLM_VERSION = None
if importlib.util.find_spec("vllm"):
    import vllm
    if hasattr(vllm, "__version_tuple__"):
        version_tuple = vllm.__version_tuple__
        CURRENT_VLLM_VERSION = ".".join([str(ele) for ele in version_tuple[:3]])
    else:
        CURRENT_VLLM_VERSION = vllm.__version__


class VLLMVersion(str, Enum):
    """support versions of vLLM."""
    v_0_8_5 = "0.8.5"


class RAY_PG_STRATEGY(Enum):
    """ray placement group strategy"""
    PACK = "PACK"
    SPREAD = "SPREAD"
