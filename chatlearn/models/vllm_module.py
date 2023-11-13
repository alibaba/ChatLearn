# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
"""RLHF VLLM module"""

import torch
from tqdm import tqdm

from vllm.config import CacheConfig, ParallelConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.engine.llm_engine import LLMEngine
from vllm.model_executor.parallel_utils import parallel_state
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata
from vllm.transformers_utils.config import get_config
from vllm.utils import get_gpu_memory, Counter
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.worker import Worker

from chatlearn.utils.logger import logger
from chatlearn.utils.vllm_utils import initialize_vllm, Megatron2TransformerSyncMap, VllmModelConfig
from .torch_module import RLHFTorchModule

# pylint: disable=import-outside-toplevel
class RLHFVLLMModule(RLHFTorchModule, LLMEngine, Worker):
    """RLHFVLLMModule is the class for RLHF Vllm models.

    Args
    ----
    name : str
        model name
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_stats = False
        self._model_config = None

        # inference only
        if self.model_args.get("micro_batch_size") != self.module_args.generation_batch_size:
            logger.info(f"{self.name} Overwrite micro_batch_size with generation_batch_size {self.module_args.generation_batch_size}")
        self.model_args["micro_batch_size"] = self.module_args.generation_batch_size

        self._init_args()

    def _init_args(self):
        self.seq_counter = Counter()
        self.request_counter = Counter()
        self.sampling_params = SamplingParams(
            n=self.model_args.get("n"),
            temperature=0.0 if self.model_args.get("use_beam_search") else 1.0,
            top_p=self.model_args.get("top_p"),
            use_beam_search=self.model_args.get("use_beam_search"),
            ignore_eos=self.model_args.get("ignore_eos"),
            max_tokens=self.model_args.get("max_new_tokens"),
            logprobs=1
        )

        self.scheduler_config = SchedulerConfig(
            self.model_args.get("max_num_batched_tokens"),
            self.model_args["micro_batch_size"],
            self.model_args.get("seq_length"),
            self.model_args.get("max_paddings"),
        )
        self.cache_config = CacheConfig(
            self.model_args.get("block_size"),
            self.model_args.get("gpu_memory_utilization"),
            self.model_args.get("swap_space"),
            None,
        )

        self.parallel_config = ParallelConfig(
            self.module_args.pipeline_model_parallel_size,
            self.module_args.tensor_model_parallel_size,
            False
        )

        self.sliding_window = self.cache_config.sliding_window

    def add_extra_args(self, parser):
        """
        Add extra arguments for vllm.

        Args
        ----
        parser : ArgumentParser
            Add extra arguments.
        """
        group = parser.add_argument_group(title='vLLM extra arguments')
        group.add_argument('--distributed-backend', default='nccl',
                           choices=['nccl', 'gloo'],
                           help='Which backend to use for distributed training.')
        group.add_argument('--distributed-timeout-minutes', type=int, default=10,
                           help='Timeout minutes for torch.distributed.')
        return parser

    def init(self):
        """
        :meta private:
        """
        self.model_args["params_dtype"] = torch.float
        if self.model_args.get("fp16", False):
            assert not self.model_args.get("bf16", False)
            self.model_args["params_dtype"] = torch.half
        if self.model_args.get("bf16", False):
            assert not self.model_args.get("fp16", False)
            self.model_args["params_dtype"] = torch.bfloat16
        self.model_args["pipeline_model_parallel_size"] = self.module_args.pipeline_model_parallel_size
        self.model_args["tensor_model_parallel_size"] = self.module_args.tensor_model_parallel_size
        initialize_vllm(extra_args_provider=self.add_extra_args,
                        ignore_unknown_args=True,
                        args_dict=self.model_args)

    def build_scheduler(self):
        self.scheduler = Scheduler(self.scheduler_config, self.cache_config)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        sampling_params = SamplingParams(top_p=0.99, top_k=self.tokenizer.vocab_size - 1)

        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_gpu_blocks, num_cpu_blocks = self.profile_num_available_blocks(sampling_params)

        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self.init_cache_engine(cache_config=self.cache_config)
        logger.info("success to call init_cache_engine")

    def _add_request_internal(self, prompt_list, prompt_token_id_list):
        for prompt, prompt_token_ids in zip(prompt_list, prompt_token_id_list):
            request_id = next(self.request_counter)
            self.add_request(
                request_id,
                prompt,
                self.sampling_params,
                prompt_token_ids=prompt_token_ids
            )
        self.outputs = []
        self.num_requests = self.get_num_unfinished_requests()
        self.pbar = tqdm(total=self.num_requests, desc="Processed prompts")
        return "ok"

    def model_setup(self):
        """
        :meta private:
        """
        super().model_setup()
        # TODO: we may need to let setup return model, optimizer and opt_param_scheduler
        if self.trainable:
            assert hasattr(self, "model")
            assert hasattr(self, "optimizer")
            assert hasattr(self, "opt_param_scheduler")
        else:
            assert hasattr(self, "model")
            self.model.eval()

        logger.info("start to init cache")
        self._init_cache()

    def map_src_to_dst(self, src_names):
        """
        :meta private:
        """
        # TODO(jiang.jl): compatible with other models.
        sync_map = Megatron2TransformerSyncMap(src_names)
        return sync_map.dst_names

    def profile_num_available_blocks(self, sampler_config):
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.

        # Enable top-k sampling to reflect the accurate memory usage.
        max_prompt_length = (
            self.model_args.get("seq_length") - self.model_args.get("max_new_tokens")
        )
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = min(self.scheduler_config.max_num_seqs, max_num_batched_tokens // max_prompt_length)
        seqs = []

        for group_id in range(max_num_seqs):
            seq_data = SequenceData([0] * max_prompt_length)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampler_config,
                block_tables=None,
            )
            seqs.append(seq)

        input_tokens, input_positions, input_metadata = self._prepare_inputs(seqs)

        # Execute the model.
        num_layers = self.num_layers() // self.pipeline_model_parallel_size()

        data = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches" : [(None, None)] * num_layers,
            "input_metadata": input_metadata,
            "cache_events":None,
        }
        self.eval_forward(data)

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        cache_block_size = CacheEngine.get_cache_block_size(
            self.cache_config.block_size, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        torch.cuda.empty_cache()

        return num_gpu_blocks, num_cpu_blocks

    @property
    def model_config(self):
        """
        :meta private:
        """
        if self._model_config is None:
            hf_config = get_config(self.model_args.get("tokenizer"), False, None)
            hf_config.torch_dtype = self.model_args.get("params_dtype")
            self._model_config = VllmModelConfig(hf_config)
        return self._model_config

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
        return None

    @property
    def data_parallel_rank(self):
        """
        :meta private:
        """
        return None

    def tensor_parallel_rank(self):
        """
        :meta private:
        """
        return parallel_state.get_tensor_model_parallel_rank()

    def pipeline_parallel_rank(self):
        """
        :meta private:
        """
        return parallel_state.get_pipeline_model_parallel_rank()

    def num_layers(self):
        """
        :meta private:
        """
        return self.model_config.hf_config.num_hidden_layers

    def schedule(self):
        seq_group_metadata_list, self.scheduler_outputs, self.ignored = self._schedule()
        if self.scheduler_outputs.is_empty():
            return self.ignored

        return {
            "seq_group_metadata_list" : seq_group_metadata_list,
            "blocks_to_swap_in" : self.scheduler_outputs.blocks_to_swap_in,
            "blocks_to_swap_out" : self.scheduler_outputs.blocks_to_swap_out,
            "blocks_to_copy" : self.scheduler_outputs.blocks_to_copy
        }

    def process_model_outputs(self, output):
        step_outputs = self._process_model_outputs(output, self.scheduler_outputs) + self.ignored
        done = 0

        for out in step_outputs:
            if out.finished:
                self.outputs.append(out)
                done += 1
                self.pbar.update(1)

        self.num_requests -= done
        if self.num_requests <= 0:
            self.pbar.close()

        return "ok"

    def execute_step(self, seq_group_metadata_list, blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy):

        output = self.execute_model(
            seq_group_metadata_list,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
        )

        if hasattr(self, "scheduler_outputs"):
            return self.process_model_outputs(output)

        return output

    def decode(self):
        self.outputs = sorted(self.outputs, key=lambda x: int(x.request_id))
        return self.decode_internal(self.outputs)
