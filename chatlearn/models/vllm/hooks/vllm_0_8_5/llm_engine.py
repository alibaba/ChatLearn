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
"""Hooks of vllm-0.8.5 llm_engine remove __reduce__ function."""

import inspect
from typing import Dict, Optional

# pylint: disable=unused-import,wildcard-import,unused-argument,wrong-import-order
from chatlearn.utils.vllm_utils import vllm_use_v1
from vllm.engine.metrics_types import StatLoggerBase
from vllm.usage.usage_lib import UsageContext
if vllm_use_v1():
    from vllm.v1.engine import llm_engine
else:
    from vllm.engine import llm_engine
    source = inspect.getsource(llm_engine.LLMEngine.__reduce__)
    if 'RuntimeError' in source:
        def __reduce__(self):
            # This is to ensure that the LLMEngine can be referenced in
            # the closure used to initialize Ray worker actors
            pass

        del llm_engine.LLMEngine.__reduce__


@classmethod
def from_engine_args(
    cls,
    engine_args,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
) -> "LLMEngine":
    """Creates an LLM engine from the engine arguments."""
    # Create the engine configs.
    engine_config = engine_args.create_engine_config(usage_context)
    if vllm_use_v1():
        from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor # pylint: disable=import-outside-toplevel
        executor_class = RayDistributedExecutor
    else:
        from vllm.executor.ray_distributed_executor import RayDistributedExecutor # pylint: disable=import-outside-toplevel
        executor_class = RayDistributedExecutor
    # Create the LLM engine.
    engine = cls( # pylint: disable=not-callable
        vllm_config=engine_config,
        executor_class=executor_class,
        log_stats=not engine_args.disable_log_stats,
        usage_context=usage_context,
        stat_loggers=stat_loggers,
    )

    return engine
llm_engine.LLMEngine.from_engine_args = from_engine_args
