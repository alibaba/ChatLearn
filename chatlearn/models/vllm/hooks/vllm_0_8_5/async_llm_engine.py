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
"""del init_ray_cluster in AsyncLLMEngine."""

from typing import Dict, Optional

# pylint: disable=unused-import,wildcard-import,unused-argument,not-callable
from vllm.config import VllmConfig
from vllm.engine import async_llm_engine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.usage.usage_lib import UsageContext

@classmethod
def from_engine_args(cls,
                    engine_args: AsyncEngineArgs,
                    engine_config: Optional[VllmConfig] = None,
                    start_engine_loop: bool = True,
                    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
                    stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
                    ) -> "AsyncLLMEngine":
    """Creates an async LLM engine from the engine arguments."""
    # Create the engine configs.
    if engine_config is None:
        engine_config = engine_args.create_engine_config(usage_context)

    executor_class = cls._get_executor_cls(engine_config)

    # Create the async LLM engine.
    engine = cls(
        vllm_config=engine_config,
        executor_class=executor_class,
        log_requests=not engine_args.disable_log_requests,
        log_stats=not engine_args.disable_log_stats,
        start_engine_loop=start_engine_loop,
        usage_context=usage_context,
        stat_loggers=stat_loggers,
    )
    return engine

async_llm_engine.AsyncLLMEngine.from_engine_args = from_engine_args
