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
"""Hooks of vllm-0.5.1 llm_engine remove __reduce__ function."""

import inspect

# pylint: disable=unused-import,wildcard-import,unused-argument
from vllm.engine import llm_engine


# source = inspect.getsource(llm_engine.LLMEngine.__reduce__)
# if 'RuntimeError' in source:
#     def __reduce__(self):
#         # This is to ensure that the LLMEngine can be referenced in
#         # the closure used to initialize Ray worker actors
#         pass

#     del llm_engine.LLMEngine.__reduce__

def _get_executor_cls(*args, **kwargs):
    # distributed_executor_backend = (
    #         engine_config.parallel_config.distributed_executor_backend)
    # assert distributed_executor_backend == "ray"
    from vllm.executor.ray_gpu_executor import RayGPUExecutor
    return RayGPUExecutor

llm_engine.LLMEngine._get_executor_cls = _get_executor_cls


    

