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
"""VLLM module v2"""

import asyncio
import inspect
import os
import sys
from typing import Optional

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import EngineConfig
from vllm.executor.ray_utils import RayWorkerWrapper
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser

from chatlearn.utils.global_vars import set_vllm_actors
from .torch_module import TorchModule


class VLLMModuleV2(TorchModule):
    """VLLMModuleV2"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_rank = 0
        os.environ['LOCAL_RANK'] = '0'
        if 'worker_module_name' in kwargs and 'worker_class_name' in kwargs:
            RayWorkerWrapper.__init__(self, **kwargs) # pylint: disable=non-parent-init-called
        os.environ['VLLM_HOST_IP'] = self.get_address()
        self.engine = None
        self.tokenizer = None

    def setup(self):
        """Set up model and load checkpoint"""
        super().setup()
        tokenizer = AutoTokenizer.from_pretrained(self.model_args['tokenizer'])
        tokenizer.tokenizer = tokenizer
        self.tokenizer = tokenizer

    def setup_vllm(self, workers):
        # setup vllm engine in rank 0
        os.environ['VLLM_HOST_IP'] = self.get_address()
        set_vllm_actors(workers)
        parser = FlexibleArgumentParser()
        parser = AsyncEngineArgs.add_cli_args(parser)
        backup_sys_argv = sys.argv
        dtype = self.model_args.get("dtype", "bfloat16")
        if self.model_args.get("fp16", False):
            dtype = "float16"
        vllm_sys_argv = ["",
                         f"--model={self.model_args['load']}",
                         f"--tensor_parallel_size={self.module_args.tensor_model_parallel_size}",
                         f"--pipeline_parallel_size={self.module_args.pipeline_model_parallel_size}",
                         f"--dtype={dtype}",
                         "--worker_use_ray",
                         "--disable_custom_all_reduce"]
        sys.argv = vllm_sys_argv
        args = parser.parse_args()
        engine_args = AsyncEngineArgs.from_cli_args(args)
        self.engine = self.from_engine_args(engine_args)

        sys.argv = backup_sys_argv
        self.tokenizer = self.engine.engine.tokenizer

    def from_engine_args(
            self,
            engine_args: AsyncEngineArgs,
            engine_config: Optional[EngineConfig] = None,
            start_engine_loop: bool = True,
            usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
            stat_loggers = None,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.

        if engine_config is None:
            engine_config = engine_args.create_engine_config()

        executor_class = AsyncLLMEngine._get_executor_cls(engine_config)

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

    def _get_sampling_params(self, is_eval):
        temperature = 0.0
        if not self.model_args.get("use_beam_search"):
            temperature = self.model_args.get("eval_temperature", 1.0) if is_eval else self.model_args.get(
                "temperature", 1.0)
        top_p = self.model_args.get("eval_top_p", 1.0) if is_eval else self.model_args.get("top_p", 1.0)
        top_k = self.model_args.get("eval_top_k", -1) if is_eval else self.model_args.get("top_k", -1)
        presence_penalty = self.model_args.get("eval_presence_penalty", 0.0) if is_eval else self.model_args.get(
            "presence_penalty", 0.0)
        frequency_penalty = self.model_args.get("eval_frequency_penalty", 0.0) if is_eval else self.model_args.get(
            "frequency_penalty", 0.0)
        repetition_penalty = self.model_args.get("eval_repetition_penalty", 1.0) if is_eval else self.model_args.get(
            "repetition_penalty", 1.0)
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
            sampling_params.use_beam_search = self.model_args.get("use_beam_search")
        return sampling_params

    async def generate_vllm(self, query, is_eval):
        prompts = query['prompt']
        seq_len = self.model_args.get("seq_length")
        final_outputs = []
        tasks = []
        for i, prompt in enumerate(prompts):
            request_id = i
            if 'sampling_param' in query:
                sampling_param = query['sampling_param'][i]
            else:
                sampling_param = self._get_sampling_params(is_eval)
                if not self.model_args.get("new_token_limit", False):
                    prompt_token_ids = query['input_ids'][i]
                    max_tokens = seq_len - len(prompt_token_ids)
                else:
                    max_tokens = self.model_args.get("max_new_tokens")
                    assert max_tokens < seq_len, "max_new_tokens must less than seq length."
                sampling_param.max_tokens = max_tokens
            task = asyncio.create_task(self.generate_one_sample(prompt, sampling_param, request_id))
            tasks.append(task)
        outputs = await asyncio.gather(*tasks)
        final_outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return final_outputs

    def is_last_rank(self):
        return True


class VLLMWokerWrapper(TorchModule, RayWorkerWrapper):
    """VLLMModule is the class for vLLM models.

    Args
    ----
    name : str
        model name
    """

    def __init__(self, *args, **kwargs):
        # avoid overwrite methods
        methods_class1 = {method[0] for method in inspect.getmembers(TorchModule, predicate=inspect.isfunction)}
        methods_class2 = {method[0] for method in inspect.getmembers(RayWorkerWrapper, predicate=inspect.isfunction)}
        common_methods = methods_class1.intersection(methods_class2)
        # common method is '__init__'
        assert common_methods == {'__init__'}, \
            f"Expected only '__init__' as common method for TorchModule and RayWorkerWrapper, but got {common_methods}"
        TorchModule.__init__(self, *args)
        self.local_rank = 0
        os.environ['LOCAL_RANK'] = '0'
        if 'worker_module_name' in kwargs and 'worker_class_name' in kwargs:
            RayWorkerWrapper.__init__(self, **kwargs) # pylint: disable=non-parent-init-called
        os.environ['VLLM_HOST_IP'] = self.get_address()
        self.engine = None

    def peak_memory(self):
        """
        :meta private:
        """
        self._peak_memory = max(self._peak_memory, torch.cuda.max_memory_allocated() / (1024 ** 3))
        return self._peak_memory
