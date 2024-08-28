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
"""Hooks of vllm-0.5.1 logits_processor to allgather logits of all ranks."""

import inspect

# pylint: disable=wildcard-import,ungrouped-imports
from vllm.model_executor.layers import logits_processor


source = inspect.getsource(logits_processor.LogitsProcessor._get_logits)
if 'tensor_model_parallel_all_gather' not in source:
    import torch
    from typing import Optional
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    def _get_logits(self, hidden_states: torch.Tensor,
                    lm_head: VocabParallelEmbedding,
                    embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
        # Get the logits for the next tokens.
        logits = lm_head.linear_method.apply(lm_head,
                                             hidden_states,
                                             bias=embedding_bias)
        from vllm.distributed.communication_op import tensor_model_parallel_all_gather # pylint: disable=import-outside-toplevel
        logits = tensor_model_parallel_all_gather(logits)
        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[:, :self.org_vocab_size]
        return logits

    logits_processor.LogitsProcessor._get_logits = _get_logits
