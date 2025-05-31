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
"""Hooks of vllm-0.8.5 logits_processor to allgather logits of all ranks."""

import inspect

# pylint: disable=wildcard-import,ungrouped-imports
from vllm.model_executor.layers import logits_processor


source = inspect.getsource(logits_processor.LogitsProcessor._gather_logits)
if 'tensor_model_parallel_gather' in source:
    import torch
    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor: # pylint: disable=unused-argument
        from vllm.distributed import tensor_model_parallel_all_gather # pylint: disable=import-outside-toplevel
        return tensor_model_parallel_all_gather(logits)

    logits_processor.LogitsProcessor._gather_logits = _gather_logits
