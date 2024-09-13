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

# LoRA
LORA_WEIGHT_PREFIX = "lora"
LORA_LAYER = "ColumnParallelLinear,Embedding,LinearLayer,RowParallelLinear,VocabParallelEmbedding"
QKV_LAYER_NAME = ["query_key_value"]


# vLLM version
CURRENT_VLLM_VERSION = None
if importlib.util.find_spec("vllm"):
    import vllm
    CURRENT_VLLM_VERSION = vllm.__version__

class VLLMVersion(Enum):
    """support versions of vLLM."""
    v_0_3_0 = "0.3.0"
    v_0_5_1 = "0.5.1"


class QwenVersion(Enum):
    """qwen version"""
    v_1 = 1.0
    v_2 = 2.0


class RAY_PG_STRATEGY(Enum):
    """ray placement group strategy"""
    PACK = "PACK"
    SPREAD = "SPREAD"

class PARAM_SYNC_COMM_TYPE(str, Enum):
    """parameter sync communication type"""
    BROADCAST = "broadcast"
    P2P = "p2p"
