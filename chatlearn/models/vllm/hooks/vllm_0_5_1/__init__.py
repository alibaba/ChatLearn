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
"""Additional hooks of vllm-0.5.1."""

from ... import is_vllm_v2

assert not is_vllm_v2(), "vLLM-0.5.1 only supports vLLM Module v1. Set env `ENABLE_VLLM_V2=False`."

from . import llm_engine
from . import logits_processor
from . import worker
