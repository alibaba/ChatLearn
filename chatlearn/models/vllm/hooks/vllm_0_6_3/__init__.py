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
"""Additional hooks of vllm-0.6.3."""

from ... import is_vllm_v2
from . import format_device_name
from . import input_preprocess

if is_vllm_v2():
    from . import async_llm_engine
    from . import llm
    from . import loader
    from . import ray_gpu_executor
    from . import worker_base
else:
    from . import llm_engine
    from . import logits_processor
