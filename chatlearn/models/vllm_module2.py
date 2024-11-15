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
"""VLLM module"""

import argparse
import gc
from typing import List, Tuple
import math
import time
import torch
from tqdm import tqdm
from typing import (Any, AsyncGenerator, Callable, Coroutine, Dict, Iterable,
                    List, Mapping, Optional, Set, Tuple, Type, Union, overload)

from chatlearn.models.vllm.vllm_model import VLLMModel
from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion
from chatlearn.utils.dist_utils import broadcast_var_object_dict
import asyncio
import json
import ssl
from argparse import Namespace
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (FlexibleArgumentParser, iterate_with_cancellation,
                        random_uuid)
from vllm.version import __version__ as VLLM_VERSION
from vllm.config import (DecodingConfig, EngineConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_timeout import asyncio_timeout
from vllm.engine.llm_engine import LLMEngine, SchedulerOutputState
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.protocol import EngineClient
from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutorAsync
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding import (
    get_guided_decoding_logits_processor)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import ExecuteModelRequest
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import deprecate_kwargs, weak_bind
from .torch_module import TorchModule
try:
    from .megatron.memory_manager import InferenceMemoryManager
except ImportError:
    InferenceMemoryManager = None
_LOGGING_INTERVAL_SEC = 5.0
from vllm.executor.ray_utils import RayWorkerWrapper

from transformers import AutoTokenizer
import os
import argparse
import gc
from typing import List, Tuple
import math
import time
import torch
from tqdm import tqdm
from typing import (Any, AsyncGenerator, Callable, Coroutine, Dict, Iterable,
                    List, Mapping, Optional, Set, Tuple, Type, Union, overload)

from chatlearn.models.vllm.vllm_model import VLLMModel
from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion
from chatlearn.utils.dist_utils import broadcast_var_object_dict
import asyncio
import json
import ssl
from argparse import Namespace
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (FlexibleArgumentParser, iterate_with_cancellation,
                        random_uuid)
from vllm.version import __version__ as VLLM_VERSION
from vllm.config import (DecodingConfig, EngineConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_timeout import asyncio_timeout
from vllm.engine.llm_engine import LLMEngine, SchedulerOutputState
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.protocol import EngineClient
from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutorAsync
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding import (
    get_guided_decoding_logits_processor)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import ExecuteModelRequest
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import deprecate_kwargs, weak_bind
import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
import torch.nn.functional as F

# pylint: disable=import-outside-toplevel,unexpected-keyword-arg,no-value-for-parameter,too-many-function-args
class VLLMModule2(TorchModule, RayWorkerWrapper):
    """VLLMModule is the class for vLLM models.

    Args
    ----
    name : str
        model name
    """

    def __init__(self, *args, **kwargs):
        TorchModule.__init__(self, *args)
        self.local_rank = 0
        os.environ['LOCAL_RANK'] = '0'
        if 'worker_module_name' in kwargs and 'worker_class_name' in kwargs:
            RayWorkerWrapper.__init__(self, **kwargs)
        self.log_stats = False

        # inference only
        if self.model_args.get("micro_batch_size") != self.module_args.generation_batch_size:
            self._logger.info(f"{self.name} Overwrite micro_batch_size with generation_batch_size {self.module_args.generation_batch_size}")
        self.model_args["micro_batch_size"] = self.module_args.generation_batch_size

        # parallel size
        self.model_args["pipeline_model_parallel_size"] = self.module_args.pipeline_model_parallel_size
        self.model_args["tensor_model_parallel_size"] = self.module_args.tensor_model_parallel_size

        # precision
        if self.model_args.get("fp16", False):
            assert not self.model_args.get("bf16", False)
            self.model_args["params_dtype"] = torch.half
        if self.model_args.get("bf16", False):
            assert not self.model_args.get("fp16", False)
            self.model_args["params_dtype"] = torch.bfloat16

        # To save gpu memory, we set `prompt_logprobs=None` default. If need to evaluate loss on prompts, please set prompt_logprobs=1
        if self.model_args.get("loss_on_prompts", False) and self.model_args.get("prompt_logprobs", None) is None:
            raise RuntimeError("expect loss_on_prompts to be false for memory reduction, or set prompt_logprobs in sampling_params to be `1`.")

        self.scheduler = None
        self._need_to_reset_scheduler = True
        self._log_metrics = self.model_args.get("log_metrics", False)
        os.environ['VLLM_HOST_IP'] = self.get_address()
        self.engine = None

    def _init_args(self):
        engine_args = EngineArgs(
            model=self.model_args.get("tokenizer"),
            tokenizer=self.model_args.get("tokenizer"),
            tokenizer_mode=self.model_args.get("tokenizer_mode", "auto"),
            trust_remote_code=self.model_args.get("trust_remote_code", True),
            tensor_parallel_size=self.module_args.tensor_model_parallel_size,
            pipeline_parallel_size=self.module_args.pipeline_model_parallel_size,
            dtype=self.model_args.get("params_dtype", "auto"),
            quantization=self.model_args.get("quantization", None),
            revision=self.model_args.get("revision", None),
            tokenizer_revision=self.model_args.get("tokenizer_revision", None),
            seed=self.model_args.get("seed", 0),
            gpu_memory_utilization=self.model_args.get("gpu_memory_utilization", 0.90),
            block_size=self.model_args.get("block_size"),
            swap_space=self.model_args.get("swap_space"),
            max_num_batched_tokens=self.model_args.get("max_num_batched_tokens"),
            max_num_seqs=self.model_args.get("micro_batch_size"),
            max_model_len=self.model_args.get("seq_length"),
            enforce_eager=True,
            disable_custom_all_reduce=True
        )

        self.quant_config = None
        if CURRENT_VLLM_VERSION == VLLMVersion.v_0_3_0:
            engine_args.max_paddings = self.model_args.get("max_paddings", 256)
            engine_args.max_context_len_to_capture = self.model_args.get("max_context_len_to_capture", 8192)
            self.model_config, self.cache_config, self.parallel_config, self.scheduler_config, self.lora_config = \
                engine_args.create_engine_configs()
            self.worker = Worker(
                self.model_config,
                self.parallel_config,
                self.scheduler_config,
                local_rank=0,
                rank=0,
                distributed_init_method=None,
                lora_config=self.lora_config,
                kv_cache_dtype=self.cache_config.cache_dtype,
                is_driver_worker=True,
            )
            self._init_tokenizer()
        elif CURRENT_VLLM_VERSION in [VLLMVersion.v_0_5_1, VLLMVersion.v_0_6_3]:
            engine_args.max_seq_len_to_capture = self.model_args.get("max_context_len_to_capture", 8192)
            engine_config = \
                engine_args.create_engine_config()
            self.cache_config = engine_config.cache_config
            self.device_config = engine_config.device_config
            self.load_config = engine_config.load_config
            self.lora_config = engine_config.lora_config
            self.model_config = engine_config.model_config
            self.parallel_config = engine_config.parallel_config
            self.scheduler_config = engine_config.scheduler_config

            self.generation_config_fields = _load_generation_config_dict(
                self.model_config)
            self.input_processor = INPUT_REGISTRY.create_input_processor(
                self.model_config)

            self.worker = Worker(
                self.model_config,
                self.parallel_config,
                self.scheduler_config,
                self.device_config,
                self.cache_config,
                self.load_config,
                local_rank=0,
                rank=0,
                distributed_init_method=None,
                lora_config=self.lora_config,
                is_driver_worker=True,
            )
            self.tokenizer = self._init_tokenizer()
            self.detokenizer = Detokenizer(self.tokenizer)

    def setup(self):
        """Set up model and load checkpoint"""
        tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/jiangle.jl/checkpoint/Llama-2-7b-hf/")
        tokenizer.tokenizer = tokenizer
        self.tokenizer = tokenizer
        # need_load_ckpt = self.src_parameter_model is None
        # model = [get_model(self.model_provider, self.model_args, need_load_ckpt)]

        # if self.model_args["load"] is None and need_load_ckpt:
        #     print_rank_0(f"Warning: Using random parameter for {self.name} model.")

        # assert len(model) == 1, "Above condition should have caught this"
        # self.model = model[0]
        # parser = FlexibleArgumentParser()
        # parser = AsyncEngineArgs.add_cli_args(parser)
        # import sys
        # backup_sys_argv = sys.argv
        # vllm_sys_argv = [""]
        # vllm_sys_argv.append(f"--model=/mnt/cworkspace/jiangle.jl/checkpoint/Llama-2-7b-hf/")
        # vllm_sys_argv.append(f"--tensor_parallel_size={self.module_args.tensor_model_parallel_size}")
        # vllm_sys_argv.append(f"--pipeline_parallel_size={self.module_args.pipeline_model_parallel_size}")
        # sys.argv = vllm_sys_argv
        # args = parser.parse_args()
        # engine_args = AsyncEngineArgs.from_cli_args(args)
        # sys.argv = backup_sys_argv
        # self.engine =  self.from_engine_args(engine_args)
        # self.tokenizer = self.engine.engine.tokenizer
        # self.vllm_engine_initialized = False

    def setup_vllm(self, workers):
        # setup vllm engine in rank 0
        os.environ['VLLM_HOST_IP'] = self.get_address()
        from chatlearn.utils.global_vars import set_vllm_actors
        set_vllm_actors(workers)
        parser = FlexibleArgumentParser()
        use_async = True
        if not use_async:
            parser = EngineArgs.add_cli_args(parser)
        else:
            parser = AsyncEngineArgs.add_cli_args(parser)
        import sys
        # os.environ['VLLM_USE_RAY_COMPILED_DAG'] = '1'
        # os.environ['VLLM_USE_RAY_SPMD_WORKER'] = '1'
        backup_sys_argv = sys.argv
        vllm_sys_argv = [""]
        vllm_sys_argv.append(f"--model=/mnt/workspace/jiangle.jl/checkpoint/Llama-2-7b-hf/")
        vllm_sys_argv.append(f"--tensor_parallel_size=8")
        vllm_sys_argv.append(f"--pipeline_parallel_size=1")
        vllm_sys_argv.append(f"--worker_use_ray")
        vllm_sys_argv.append(f"--disable_custom_all_reduce")
        sys.argv = vllm_sys_argv
        args = parser.parse_args()
        # self.model_args = self.module_args.args_dict
        if not use_async:
            engine_args = EngineArgs.from_cli_args(args)
            self.engine =  LLMEngine.from_engine_args(engine_args)
        else:
            engine_args = AsyncEngineArgs.from_cli_args(args)
            self.engine =  self.from_engine_args(engine_args)

        sys.argv = backup_sys_argv
        self.tokenizer = self.engine.engine.tokenizer
        #from chatlearn.models.vllm.vllm_engine import VllmEngine
        # self.engine = VllmEngine().engine
        # event_loop = asyncio.get_running_loop()
        # self.vllm_engine_initialized = True
    
    def from_engine_args(
        self,
        engine_args: EngineArgs,
        engine_config: Optional[EngineConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.

        if engine_config is None:
            engine_config = engine_args.create_engine_config()

        executor_class = AsyncLLMEngine._get_executor_cls(engine_config)

        # if executor_class.uses_ray:
        #     initialize_ray_cluster(engine_config.parallel_config)
        # Create the async LLM engine.
        engine = AsyncLLMEngine(
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )
        return engine
    
    async def generate_one_sample(self, prompt, sampling_param, request_id):
        results_generator = self.engine.generate(prompt, sampling_param, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        return final_output


    def reinit_cache_engine(self):
        # reinit cache engine
        if CURRENT_VLLM_VERSION == VLLMVersion.v_0_3_0:
            self.worker.init_cache_engine(cache_config=self.cache_config)
            self.worker.warm_up_model()
        elif CURRENT_VLLM_VERSION in [VLLMVersion.v_0_5_1, VLLMVersion.v_0_6_3]:
            self.worker.initialize_cache(self.cache_config.num_gpu_blocks, self.cache_config.num_cpu_blocks)

    def empty_cache(self):
        if CURRENT_VLLM_VERSION == VLLMVersion.v_0_3_0:
            self.worker.gpu_cache = None # pylint: disable=access-member-before-definition
            self.worker.cache_engine.cpu_cache = None
            self.worker.cache_engine.gpu_cache = None
        elif CURRENT_VLLM_VERSION in [VLLMVersion.v_0_5_1, VLLMVersion.v_0_6_3]:
            if self.worker.gpu_cache is not None:
                for ele in self.worker.gpu_cache: # pylint: disable=unused-variable
                    ele = None
                self.worker.gpu_cache = None # pylint: disable=access-member-before-definition

            if hasattr(self.worker, "cache_engine") and self.worker.cache_engine is not None:
                for c_e in self.worker.cache_engine:
                    c_e.cpu_cache = None
                    c_e.gpu_cache = None
                self.worker.cache_engine = None

        self.clear_cache()

    def clear_cache(self):
        if not self.timers("gc").started_:
            self.timers("gc").start()
        gc.collect()
        self.timers("gc").stop()

        super().empty_cache()

    def _add_request(self, data, is_eval=False): # pylint: disable=arguments-differ
        prompt_key = self.model_args.get("vllm_prompt_key", "prompt")
        input_ids_key = self.model_args.get("vllm_input_ids_key", "input_ids")
        return self._add_request_internal(data[prompt_key], data[input_ids_key], is_eval=is_eval)

    def _get_sampling_params(self, is_eval):
        temperature = 0.0
        if not self.model_args.get("use_beam_search"):
            temperature = self.model_args.get("eval_temperature", 1.0) if is_eval else self.model_args.get("temperature", 1.0)
        top_p = self.model_args.get("eval_top_p", 1.0) if is_eval else self.model_args.get("top_p", 1.0)
        top_k = self.model_args.get("eval_top_k", -1) if is_eval else self.model_args.get("top_k", -1)
        presence_penalty = self.model_args.get("eval_presence_penalty", 0.0) if is_eval else self.model_args.get("presence_penalty", 0.0)
        frequency_penalty = self.model_args.get("eval_frequency_penalty", 0.0) if is_eval else self.model_args.get("frequency_penalty", 0.0)
        repetition_penalty = self.model_args.get("eval_repetition_penalty", 1.0) if is_eval else self.model_args.get("repetition_penalty", 1.0)
        stop = self.model_args.get("stop_token_list", None)
        if isinstance(stop, str):
            stop = stop.split(";")
        sampling_params = SamplingParams(
            n=self.model_args.get("n"),
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ignore_eos=self.model_args.get("ignore_eos"),
            stop=stop,
            logprobs=1,
            prompt_logprobs=self.model_args.get("prompt_logprobs", None),
            skip_special_tokens=False
        )
        # VLLMVersion.v_0_3_0, VLLMVersion.v_0_5_1
        if hasattr(sampling_params, 'use_beam_search'):
            sampling_params.use_beam_search=self.model_args.get("use_beam_search")
        return sampling_params

    def _add_request_internal(self, prompt_list, prompt_token_id_list, is_eval=False):
        if self._need_to_reset_scheduler:
            self._reset_scheduler()
        self.reset_vllm()

        # sampling params
        temperature = 0.0
        if not self.model_args.get("use_beam_search"):
            temperature = self.model_args.get("eval_temperature", 1.0) if is_eval else self.model_args.get("temperature", 1.0)
        top_p = self.model_args.get("eval_top_p", 1.0) if is_eval else self.model_args.get("top_p", 1.0)
        top_k = self.model_args.get("eval_top_k", -1) if is_eval else self.model_args.get("top_k", -1)
        presence_penalty = self.model_args.get("eval_presence_penalty", 0.0) if is_eval else self.model_args.get("presence_penalty", 0.0)
        frequency_penalty = self.model_args.get("eval_frequency_penalty", 0.0) if is_eval else self.model_args.get("frequency_penalty", 0.0)
        repetition_penalty = self.model_args.get("eval_repetition_penalty", 1.0) if is_eval else self.model_args.get("repetition_penalty", 1.0)

        stop = self.model_args.get("stop_token_list", None)
        if isinstance(stop, str):
            stop = stop.split(";")
        seq_len = self.model_args.get("seq_length")
        for prompt, prompt_token_ids in zip(prompt_list, prompt_token_id_list):
            request_id = next(self.request_counter)
            if self.model_args.get("new_token_limit", False):
                max_tokens = self.model_args.get("max_new_tokens")
                assert max_tokens < seq_len, "max_new_tokens must less than seq length."
                prompt_token_ids = prompt_token_ids \
                    if len(prompt_token_ids) <= seq_len-max_tokens \
                    else prompt_token_ids[:seq_len-max_tokens]
            else:
                if len(prompt_token_ids) >= seq_len:
                    prompt_token_ids = prompt_token_ids[:seq_len-1]
                max_tokens = seq_len - len(prompt_token_ids)
            
            if CURRENT_VLLM_VERSION in [VLLMVersion.v_0_3_0, VLLMVersion.v_0_5_1]:
                sampling_params = SamplingParams(
                    n=self.model_args.get("n"),
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    use_beam_search=self.model_args.get("use_beam_search"),
                    ignore_eos=self.model_args.get("ignore_eos"),
                    stop=stop,
                    max_tokens=max_tokens,
                    logprobs=1,
                    prompt_logprobs=self.model_args.get("prompt_logprobs", None),
                    skip_special_tokens=False
                )
            elif CURRENT_VLLM_VERSION == VLLMVersion.v_0_6_3:
                sampling_params = SamplingParams(
                    n=self.model_args.get("n"),
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    ignore_eos=self.model_args.get("ignore_eos"),
                    stop=stop,
                    max_tokens=max_tokens,
                    logprobs=1,
                    prompt_logprobs=self.model_args.get("prompt_logprobs", None),
                    skip_special_tokens=False
                )
            else:
                raise RuntimeError(f"Unsupported vllm version {CURRENT_VLLM_VERSION}, expect one of {list(VLLMVersion)}")

            if CURRENT_VLLM_VERSION == VLLMVersion.v_0_3_0:
                self.add_request(
                    request_id,
                    prompt,
                    sampling_params,
                    prompt_token_ids=prompt_token_ids
                )
            elif CURRENT_VLLM_VERSION in \
                    [VLLMVersion.v_0_5_1, VLLMVersion.v_0_6_3]:
                inputs = self.convert_v1_inputs(
                    prompts=[prompt],
                    prompt_token_ids=[prompt_token_ids],
                )[0]
                self.add_request(
                    request_id,
                    inputs,
                    sampling_params
                )

        self.outputs = []
        self.num_requests = self.get_num_unfinished_requests()
        self._reset_metrics_stats_args()
        self.pbar = tqdm(total=self.num_requests, desc=f"Processed prompts (replica {self.replica_id+1}/{self._num_replica})")
        self._need_to_reset_scheduler = True

    def model_setup(self):
        """
        :meta private:
        """
        super().model_setup()
        # TODO: we may need to let setup return model, optimizer and opt_param_scheduler
       # if self.trainable:
       #     assert hasattr(self, "model")
       #     assert hasattr(self, "optimizer")
       #     assert hasattr(self, "opt_param_scheduler")
       #     self.model.eval()
       # else:
       #     assert hasattr(self, "model")
       #     self.model.eval()
       # self.worker.model_runner.model = self.model.model
       # if CURRENT_VLLM_VERSION in [VLLMVersion.v_0_5_1, VLLMVersion.v_0_6_3]:
       #     self.worker.device = torch.device(f"cuda:{torch.cuda.current_device()}")
       #     self.worker.init_gpu_memory = torch.cuda.mem_get_info()[0]

       # if self.module_args.offload_weights:
       #     if InferenceMemoryManager is None:
       #         raise Exception("Import InferenceMemoryManager failed, you may need to set right Megatron path first.")
       #     self._memory_manager = InferenceMemoryManager(
       #         self.model,
       #         self.runtime_args.bucket_size_mb_in_memory_manager,
       #     )
       #     self.offload()

    def pipeline_model_parallel_size(self):
        """
        get pipeline_model_parallel_size

        :meta private:
        """
        return self.parallel_config.pipeline_parallel_size

    def tensor_model_parallel_size(self):
        """
        get tensor_model_parallel_size

        :meta private:
        """
        return self.parallel_config.tensor_parallel_size

    @property
    def data_parallel_size(self):
        """
        :meta private:
        """
        return 1

    @property
    def data_parallel_rank(self):
        """
        :meta private:
        """
        return 0

    def tensor_parallel_rank(self):
        """
        :meta private:
        """
        return parallel_state.get_tensor_model_parallel_rank()

    def pipeline_parallel_rank(self):
        """
        :meta private:
        """
        return get_pipeline_model_parallel_rank()

    def num_layers(self):
        """
        :meta private:
        """
        return self.model_config.hf_config.num_hidden_layers


    async def _generate_vllm(self, query, is_eval):
        prompts = query['prompt']
        seq_len = self.model_args.get("seq_length")
        final_outputs = []
        tasks = []
        for i, prompt in enumerate(prompts):
            request_id = i
            # TODO: call it only once for is_eval and none-is_eval
            if 'sampling_param' in query:
                sampling_param = query['sampling_param'][i]
            else:
                sampling_param = self._get_sampling_params(is_eval)
                if not self.model_args.get("new_token_limit", False):
                    prompt_token_ids = query['input_ids'][i]
                    # if len(prompt_token_ids) >= seq_len:
                    #     prompt_token_ids = prompt_token_ids[:seq_len-1]
                    max_tokens = seq_len - len(prompt_token_ids)
                else:
                    max_tokens = self.model_args.get("max_new_tokens")
                    assert max_tokens < seq_len, "max_new_tokens must less than seq length."
                sampling_param.max_tokens = max_tokens
            task = asyncio.create_task(self.generate_one_sample(prompt, sampling_param, request_id))
            tasks.append(task)
        final_outputs = await asyncio.gather(*tasks)
        # logger.info(final_outputs)
        return final_outputs

    def generate_vllm(self, data, is_eval):
        import asyncio
        loop = asyncio.get_event_loop()

        outputs = loop.run_until_complete(self._generate_vllm(data, True))
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        loop.close()
        # results = self.decode_internal(results)
        return outputs

    # def generate_vllm(self, data, is_eval):
    #     # import asyncio
    #     # if self.is_first_rank():
    #     outputs = self._generate_vllm(data, is_eval)
    #     # outputs = asyncio.run(self._generate_vllm(data, is_eval))
    #     self.outputs = sorted(outputs, key=lambda x: int(x.request_id))
    #     return self.outputs


    def offload_weights(self):
        """
        offload weights
        """
        if self.module_args.offload_weights:
            self._memory_manager.offload_weights()

    def onload_weights(self):
        """
        onload weights
        """
        if self.module_args.offload_weights:
            self._memory_manager.onload_weights()
