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
""""Version compatibility for vLLM"""

from typing import List, TypedDict
from typing_extensions import NotRequired
from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion

# pylint: disable=unused-import,import-outside-toplevel,wrong-import-position,wrong-import-order

if CURRENT_VLLM_VERSION == VLLMVersion.v_0_8_5:
    from vllm.core.interfaces import BlockSpaceManager
    from vllm.distributed import parallel_state
    from vllm.distributed.communication_op import tensor_model_parallel_all_gather
    from vllm.distributed.parallel_state import init_world_group
    from vllm.distributed.parallel_state import initialize_model_parallel
    from vllm.distributed.utils import get_pp_indices
    from vllm.engine.async_llm_engine import _AsyncLLMEngine as LLMEngine
    from vllm.engine.llm_engine import SchedulerContext, SchedulerOutputState
    from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
    from vllm.engine.output_processor.stop_checker import StopChecker
    from vllm.inputs import INPUT_REGISTRY
    from vllm.inputs.preprocess import InputPreprocessor
    from vllm.model_executor.model_loader.utils import set_default_torch_dtype as _set_default_torch_dtype
    from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM
    from vllm.sequence import ExecuteModelRequest
    from vllm.transformers_utils.detokenizer import Detokenizer


from vllm.core.scheduler import Scheduler
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.llm import LLM
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.qwen import QWenLMHeadModel
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
from vllm.worker.worker import Worker



def get_model_architecture(config):
    from vllm.model_executor.model_loader.utils import get_model_architecture  as get_model_architecture_v2
    return get_model_architecture_v2(config)[0]


def get_pipeline_model_parallel_rank():
    return parallel_state.get_pp_group().rank_in_group


def get_pipeline_model_parallel_world_size():
    return parallel_state.get_pp_group().world_size
