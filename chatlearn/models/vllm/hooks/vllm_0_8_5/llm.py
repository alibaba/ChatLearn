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
"""Hooks of vllm-0.8.5 llm init with AsyncLLMEngine and AsyncEngineArgs."""

from typing import Any, Optional, Union
import cloudpickle

# pylint: disable=unused-import,wildcard-import,unused-argument,unexpected-keyword-arg
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import AsyncEngineArgs, HfOverrides, TaskOption, PoolerConfig
from vllm.entrypoints import llm
from vllm import envs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter
from vllm.engine.llm_engine import LLMEngine

def init(
    self,
    model: str,
    tokenizer: Optional[str] = None,
    tokenizer_mode: str = "auto",
    skip_tokenizer_init: bool = False,
    trust_remote_code: bool = False,
    allowed_local_media_path: str = "",
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    quantization: Optional[str] = None,
    revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    seed: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    swap_space: float = 4,
    cpu_offload_gb: float = 0,
    enforce_eager: Optional[bool] = None,
    max_seq_len_to_capture: int = 8192,
    disable_custom_all_reduce: bool = False,
    disable_async_output_proc: bool = False,
    hf_token: Optional[Union[bool, str]] = None,
    hf_overrides: Optional[HfOverrides] = None,
    mm_processor_kwargs: Optional[dict[str, Any]] = None,
    # After positional args are removed, move this right below `model`
    task: TaskOption = "auto",
    override_pooler_config: Optional[PoolerConfig] = None,
    compilation_config: Optional[Union[int, dict[str, Any]]] = None,
    **kwargs,
) -> None:
    '''
    LLM constructor.

    Note: if enforce_eager is unset (enforce_eager is None)
    it defaults to False.
    '''

    if "disable_log_stats" not in kwargs:
        kwargs["disable_log_stats"] = True

    if "worker_cls" in kwargs:
        worker_cls = kwargs["worker_cls"]
        # if the worker_cls is not qualified string name,
        # we serialize it using cloudpickle to avoid pickling issues
        if isinstance(worker_cls, type):
            kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

    if compilation_config is not None:
        if isinstance(compilation_config, (int, dict)):
            compilation_config_instance = CompilationConfig.from_cli(
                str(compilation_config))
        else:
            compilation_config_instance = compilation_config
    else:
        compilation_config_instance = None
    ## monkey patch EngineArgs -> AsyncEngineArgs
    engine_args = AsyncEngineArgs(
        model=model,
        task=task,
        tokenizer=tokenizer,
        tokenizer_mode=tokenizer_mode,
        skip_tokenizer_init=skip_tokenizer_init,
        trust_remote_code=trust_remote_code,
        allowed_local_media_path=allowed_local_media_path,
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
        max_seq_len_to_capture=max_seq_len_to_capture,
        disable_custom_all_reduce=disable_custom_all_reduce,
        disable_async_output_proc=disable_async_output_proc,
        hf_token=hf_token,
        hf_overrides=hf_overrides,
        mm_processor_kwargs=mm_processor_kwargs,
        override_pooler_config=override_pooler_config,
        compilation_config=compilation_config_instance,
        **kwargs,
    )

    # Create the Engine (autoselects V0 vs V1)
    self.llm_engine = LLMEngine.from_engine_args(
        engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)
    self.engine_class = type(self.llm_engine)

    self.request_counter = Counter()
    self.default_sampling_params: Union[dict[str, Any], None] = None

llm.LLM.__init__ = init
