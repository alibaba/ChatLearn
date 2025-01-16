# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4
from typing import List

import torch
import uvloop
from transformers import AutoTokenizer

from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.utils import merge_async_iterators


@dataclasses.dataclass
class SampleRequest:
    """A class representing a single inference request for benchmarking.

    Attributes:
        prompt: The input text prompt for the model.
        prompt_len: The length of the prompt in tokens.
        output_len: The expected length of the output in tokens.
    """
    prompt: str
    prompt_len: int
    output_len: int

def print_and_write_to_file(file_handle, message):
    file_handle.write(message)
    print(message, flush=True)

def example_to_requests(args, tokenizer, examples):
    requests: List[SampleRequest] = []
    sampling_params: List[SamplingParams] = []

    sample = args.num_sampling

    for example in examples:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": example["system"]}, {"role": "user", "content": example["message"]}],
            # [{"role": "user", "content": ex["query"]}],
            add_generation_prompt=True,
            tokenize=False,
        )

        prompt_len = len(tokenizer(prompt).input_ids)
        if args.max_tokens is not None:
            max_tokens = args.max_tokens
        else:
            max_tokens = 2048 - len(tokenizer(prompt).input_ids)
        requests.append(SampleRequest(prompt, prompt_len, max_tokens))
        
        sampling_params.append(
            SamplingParams(
                n=sample,
                temperature=1,
                top_p=0.9,
                top_k=-1,
                ignore_eos=False,
                stop=['<|im_start|>', '<|im_end|>', '<|endoftext|>'],
                max_tokens=max_tokens,
                logprobs=1,
                skip_special_tokens=False
            )
        )
        
    return requests, sampling_params

async def run_query(args, engine, prompt: str, sampling_param: SamplingParams):
    request_id = uuid4()

    outputs = engine.generate(
        prompt,
        sampling_param,
        request_id
    )
    async for output in outputs:
        final_output = output
    responses = []
    for output in final_output.outputs:
        responses.append(output)
    return responses

async def run_vllm_async_legacy(
    args,
    engine: AsyncLLMEngine,
    requests: List[SampleRequest],
    sampling_params: List[SamplingParams],
):
    prompts = [request.prompt for request in requests]
    tasks = [asyncio.create_task(run_query(args, engine, prompt, sp)) for prompt, sp in zip(prompts, sampling_params)]
    outputs = []
    tic = time.perf_counter()
    for task in asyncio.as_completed(tasks):
        result = await task
        outputs.append(result)
    toc = time.perf_counter()
    elapsed_time = toc - tic
    return outputs, elapsed_time

async def run_vllm_async_experimental(
    args,
    engine_args: AsyncEngineArgs,
    requests: List[SampleRequest],
    sampling_params: List[SamplingParams],
    disable_frontend_multiprocessing: bool = False,
):
    prompts = [request.prompt for request in requests]
    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:
        generators = []
        tic = time.perf_counter()
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            generator = llm.generate(prompt,
                                     sp,
                                     request_id=f"test{i}")
            generators.append(generator)

        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass
        toc = time.perf_counter()
        elapsed_time = toc - tic

        return None, elapsed_time

def run_vllm_sync(args, engine:LLM, requests:List[SampleRequest], sampling_params:List[SamplingParams]):
    sample = args.num_sampling
    prompts = [request.prompt for request in requests]

    tic = time.perf_counter()
    vllm_outputs = engine.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
    )
    toc = time.perf_counter()

    outputs = [[vllm_output.outputs[i] for i in range(sample)] for vllm_output in vllm_outputs]
    elapsed_time = toc - tic

    return outputs, elapsed_time

def process_outputs(args, requests, outputs, elapsed_time):
    log_dir = args.log_dir
    model_name = args.model_name
    use_async_llm_engine = args.use_async_llm_engine
    num_prompts = args.num_prompts
    sample = args.num_sampling
    enforce_eager = args.enforce_eager
    tp_size = args.tensor_parallel_size
    pp_size = args.pipeline_parallel_size
    spmd = args.vllm_use_ray_spmd_worker
    max_tokens = args.max_tokens
    legacy = not args.experimental

    with open(
        f"{log_dir}/benchmark_{model_name}_AsyncLLMEngine{use_async_llm_engine}"
        f"_vllm_prompts{num_prompts}_sample{sample}_enforceeager{enforce_eager}"
        f"_tp{tp_size}pp{pp_size}_spmd{spmd}_max_tokens{max_tokens}_legacy{legacy}.log", "w"
    ) as file_handle:
        print_and_write_to_file(
            file_handle,
            (
                f"load {num_prompts} examples\n"
                f"total time : {elapsed_time}\n"
                f"average time per batch : {elapsed_time}"
        ))

        total_token = 0
        max_len = 0
        if outputs is None:
            max_len = 0
            for request in requests:
                total_token += request.output_len * sample
                max_len = max(max_len, request.output_len)
        else:
            for responses_for_one_prompt in outputs:
                for response in responses_for_one_prompt:
                    token_ids = response.token_ids
                    total_token += len(token_ids)
                    max_len = max(max_len, len(token_ids))
        avg_len = total_token / (num_prompts * sample)

        print_and_write_to_file(
            file_handle,
            (
                f"total output token : {total_token}\n"
                f"max output len : {max_len}\n"
                f"avg output len : {avg_len}\n"
                f"output token throuput per second : {total_token / elapsed_time}"
        ))

        if outputs is None:
            return

        print_and_write_to_file(
            file_handle,
            f"A total number of {len(outputs) * sample} outputs are generated"
        )
        s = ""
        weird_count = 0
        index = 0
        for ex in outputs:
            for iid, rsp in enumerate(ex):
                rsp_text = rsp.text
                s += f"rsp {index*32+iid} : {rsp_text}\n"
                if "same same" in rsp_text:
                    weird_count += 1
            index += 1
        print_and_write_to_file(file_handle, f"weird_count {weird_count}")
        file_handle.write(s)

def benchmark_vllm(args):
    if args.vllm_use_ray_spmd_worker:
        os.environ['VLLM_USE_RAY_SPMD_WORKER'] = '1'
        os.environ['VLLM_USE_RAY_COMPILED_DAG'] = '1'
        os.environ['VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL'] = '0'
    else:
        os.environ['VLLM_USE_RAY_SPMD_WORKER'] = '0'
        os.environ['VLLM_USE_RAY_COMPILED_DAG'] = '0'
    tp_size = args.tensor_parallel_size
    pp_size = args.pipeline_parallel_size
    enforce_eager = args.enforce_eager
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_prompts = args.num_prompts
    num_scheduler_steps = args.num_scheduler_steps

    data_file = args.data
    with open(data_file) as f:
        raw_ex_batch = [json.loads(line) for line in f]
    raw_ex_batch = raw_ex_batch[:num_prompts]
    requests, sampling_params = example_to_requests(args, tokenizer, raw_ex_batch)

    kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": tp_size,
        "dtype": torch.bfloat16,
        "enforce_eager": enforce_eager,
        "disable_custom_all_reduce": True,
        "distributed_executor_backend": "ray",
        "gpu_memory_utilization": 0.85,
        "num_scheduler_steps": num_scheduler_steps,
    }

    if args.use_async_llm_engine:
        kwargs["pipeline_parallel_size"] = pp_size

        if not args.experimental:
            engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(**kwargs)
            )
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            outputs, elapsed_time = asyncio.run(run_vllm_async_legacy(
                args, engine, requests, sampling_params
            ))
        else:
            engine_args = AsyncEngineArgs(**kwargs)
            outputs, elapsed_time = uvloop.run(
                run_vllm_async_experimental(
                    args,
                    engine_args,
                    requests,
                    sampling_params,
            ))
    else:
        engine = LLM(**kwargs)
        outputs, elapsed_time = run_vllm_sync(args, engine, requests, sampling_params)

    process_outputs(args, requests, outputs, elapsed_time)

def validate(args):
    if args.use_async_llm_engine is False:
        assert args.pipeline_parallel_size == 1, (
            "pipeline parallelism should be disabled for LLM.generate."
        )
        assert args.experimental is False, (
            "experimental benchmark method is only applicable for AsyncLLMEngine."
        )
    else:
        if args.experimental:
            print("WARNING: Enabling experimental AsyncLLMEngine benchmarking.")
    return

def main(args):
    validate(args)
    benchmark_vllm(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Benchmark for RLHF tasks.")
    parser.add_argument('--model-path', type=str, default=None, required=True)
    parser.add_argument('--model-name', type=str, default=None, required=True)
    parser.add_argument('--data', type=str, default=None, required=True)
    parser.add_argument('--log-dir', type=str, default=None, required=True)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--pipeline-parallel-size', type=int, default=1)
    parser.add_argument('--num-prompts', type=int, default=1)
    parser.add_argument('--num-sampling', type=int, default=1)
    parser.add_argument('--use-async-llm-engine', action='store_true')
    parser.add_argument('--enforce-eager', action='store_true')
    parser.add_argument('--vllm-use-ray-spmd-worker', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=None)
    parser.add_argument('--num-scheduler-steps', type=int, default=1)
    parser.add_argument('--experimental', action='store_true', help='Experimental benchmark method from `benchmark_throughput.py` in vLLM.')

    args = parser.parse_args()
    main(args)
