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
"""Hooks of vllm-0.6.3 llm init with AsyncLLMEngine and AsyncEngineArgs."""

from typing import Any, Dict, Optional

# pylint: disable=unused-import,wildcard-import,unused-argument
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints import llm
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter

def init(
    self,
    model: str,
    tokenizer: Optional[str] = None,
    tokenizer_mode: str = "auto",
    skip_tokenizer_init: bool = False,
    trust_remote_code: bool = False,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    quantization: Optional[str] = None,
    revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    seed: int = 0,
    gpu_memory_utilization: float = 0.9,
    swap_space: float = 4,
    cpu_offload_gb: float = 0,
    enforce_eager: Optional[bool] = None,
    max_context_len_to_capture: Optional[int] = None,
    max_seq_len_to_capture: int = 8192,
    disable_custom_all_reduce: bool = False,
    disable_async_output_proc: bool = False,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    '''
    LLM constructor.

    Note: if enforce_eager is unset (enforce_eager is None)
    it defaults to False.
    '''
    if "disable_log_stats" not in kwargs:
        kwargs["disable_log_stats"] = True

    engine_args = AsyncEngineArgs(
        model=model,
        tokenizer=tokenizer,
        tokenizer_mode=tokenizer_mode,
        skip_tokenizer_init=skip_tokenizer_init,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        quantization=quantization,
        revision=revision,
        tokenizer_revision=tokenizer_revision,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=swap_space,
        cpu_offload_gb=cpu_offload_gb,
        enforce_eager=enforce_eager,
        max_context_len_to_capture=max_context_len_to_capture,
        max_seq_len_to_capture=max_seq_len_to_capture,
        disable_custom_all_reduce=disable_custom_all_reduce,
        disable_async_output_proc=disable_async_output_proc,
        mm_processor_kwargs=mm_processor_kwargs,
        **kwargs,
    )

    self.llm_engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.LLM_CLASS).engine
    self.request_counter = Counter()

llm.LLM.__init__ = init
