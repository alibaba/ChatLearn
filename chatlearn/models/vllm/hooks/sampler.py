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
"""Hooks of vllm-0.3.0 sampler to allgather logits of all ranks."""

import inspect
# pylint: disable=unused-import,wildcard-import
from vllm.model_executor.layers import sampler


source = inspect.getsource(sampler.Sampler._get_logits)
if 'tensor_model_parallel_all_gather' not in source:
    import torch
    from typing import Dict, List, Optional, Tuple
    def _get_logits(self, hidden_states: torch.Tensor, embedding: torch.Tensor,
                    embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        from vllm.model_executor.parallel_utils.communication_op import tensor_model_parallel_all_gather # pylint: disable=import-outside-toplevel
        logits = tensor_model_parallel_all_gather(logits)
        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[:, :self.org_vocab_size]
        return logits

    sampler.Sampler._get_logits = _get_logits
