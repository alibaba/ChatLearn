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

import gc
from typing import List, Tuple
import math
import os
import time
import torch
from tqdm import tqdm

from chatlearn.models.vllm.vllm_model import VLLMModel
from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion
from chatlearn.utils.dist_utils import broadcast_var_object_dict
from chatlearn.utils.vllm_import_helper import get_block_manager_cls
from chatlearn.utils.vllm_import_helper import get_pipeline_model_parallel_rank
from chatlearn.utils.vllm_import_helper import Scheduler
from chatlearn.utils.vllm_import_helper import EngineArgs
from chatlearn.utils.vllm_import_helper import LLM
from chatlearn.utils.vllm_import_helper import LLMEngine
from chatlearn.utils.vllm_import_helper import parallel_state
from chatlearn.utils.vllm_import_helper import SamplingParams
from chatlearn.utils.vllm_import_helper import Counter
from chatlearn.utils.vllm_import_helper import Worker
# additional imports for vLLM-0.5.1/0.6.3
try:
    from chatlearn.utils.vllm_import_helper import Detokenizer
    from chatlearn.utils.vllm_import_helper import ExecuteModelRequest
    from chatlearn.utils.vllm_import_helper import INPUT_REGISTRY
    from chatlearn.utils.vllm_import_helper import _load_generation_config_dict
    from chatlearn.utils.vllm_import_helper import SequenceGroupOutputProcessor
    from chatlearn.utils.vllm_import_helper import StopChecker
    from chatlearn.utils.vllm_import_helper import TextTokensPrompt
except ImportError:
    print("Cannot import addtional module for vllm 0.5.1 or 0.6.3, please install vllm 0.5.1/0.6.3 first.")
# additional imports for vLLM-0.6.3
try:
    from chatlearn.utils.vllm_import_helper import InputPreprocessor
    from chatlearn.utils.vllm_import_helper import SchedulerContext, SchedulerOutputState
except ImportError:
    print("Cannot import addtional module for vllm 0.6.3, please install vllm 0.6.3 first.")

from chatlearn.utils.vllm_utils import initialize_vllm

from chatlearn.utils.vllm_utils import get_model, print_rank_0
from .torch_module import TorchModule
try:
    from .megatron.memory_manager import InferenceMemoryManager
except ImportError:
    InferenceMemoryManager = None
_LOGGING_INTERVAL_SEC = 5.0


# pylint: disable=import-outside-toplevel,unexpected-keyword-arg,no-value-for-parameter,too-many-function-args
class VLLMModule(TorchModule, LLMEngine, LLM):
    """VLLMModule is the class for vLLM models.

    Args
    ----
    name : str
        model name
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self._init_args()

    def _init_args(self):
        if CURRENT_VLLM_VERSION == VLLMVersion.v_0_6_3:
            self.set_vllm_pp_layer_partition()

        if self.model_args.get("apply_replica_id_to_seed", True):
            seed = self.model_args.get("seed", 0) + self.replica_id
        else:
            seed = self.model_args.get("seed", 0)

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
            seed=seed,
            gpu_memory_utilization=self.model_args.get("gpu_memory_utilization", 0.90),
            block_size=self.model_args.get("block_size"),
            swap_space=self.model_args.get("swap_space"),
            max_num_batched_tokens=self.model_args.get("max_num_batched_tokens"),
            max_num_seqs=self.model_args.get("micro_batch_size"),
            max_model_len=self.model_args.get("seq_length"),
            enforce_eager=self.model_args.get("enforce_eager", True),
            disable_custom_all_reduce=True
        )

        self.quant_config = None
        self.pipeline_layer_offset = None
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
            if CURRENT_VLLM_VERSION == VLLMVersion.v_0_6_3:
                engine_args.num_scheduler_steps = self.model_args.get("num_scheduler_steps", 1)
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

            if CURRENT_VLLM_VERSION == VLLMVersion.v_0_6_3 and self.scheduler_config.is_multi_step:
                from vllm.worker.multi_step_worker import MultiStepWorker
                self.worker = MultiStepWorker(
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
            else:
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
        need_load_ckpt = self.src_parameter_model is None
        model = [get_model(self.model_provider, self.model_args, need_load_ckpt)]

        if self.model_args["load"] is None and need_load_ckpt:
            print_rank_0(f"Warning: Using random parameter for {self.name} model.")

        assert len(model) == 1, "Above condition should have caught this"
        self.model = model[0]

    def model_provider(self):
        """Build the model."""
        print_rank_0('building vLLM model ...')
        model = VLLMModel(self.model_config, self.model_args, self.cache_config)

        return model

    def _reset_metrics_stats_args(self):
        self.start_time = None
        # Logging.
        self.last_stats_time = 0.0
        self.forward_count = 0
        self.num_done_requests = 0
        self.num_processed_prompt = 0
        self.num_generated_tokens = 0
        self.action_length = 0
        self.action_max_length = float("-inf")
        self.action_min_length = float("inf")
        self.batch_size_stats = 0.0
        self.gpu_cache_usage = 0.0
        self.cpu_cache_usage = 0.0
        self.max_prompt_length_static_batching = [
            0 for _ in range(math.ceil(self.num_requests/self.scheduler_config.max_num_seqs))]
        self.max_output_length_static_batching = [
            0 for _ in range(math.ceil(self.num_requests/self.scheduler_config.max_num_seqs))]

    def reset_vllm(self):
        self.request_counter = Counter()

        self.log_stats = self.model_args.get("log_stats", False)
        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []
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
        if CURRENT_VLLM_VERSION in [VLLMVersion.v_0_5_1, VLLMVersion.v_0_6_3]:
            parallel_state.set_custom_all_reduce(not self.parallel_config.disable_custom_all_reduce)
        initialize_vllm(extra_args_provider=self.add_extra_args,
                        ignore_unknown_args=True,
                        args_dict=self.model_args)
        self.parallel_config.rank = torch.distributed.get_rank()

    def build_scheduler(self):
        self.seq_counter = Counter()
        if CURRENT_VLLM_VERSION == VLLMVersion.v_0_3_0:
            if self.scheduler is None:
                self.scheduler = Scheduler(self.scheduler_config, self.cache_config, None)
            else:
                BlockSpaceManagerImpl = get_block_manager_cls(None)
                self.scheduler.block_manager = BlockSpaceManagerImpl( # pylint: disable=abstract-class-instantiated
                    block_size=self.cache_config.block_size,
                    num_gpu_blocks=self.cache_config.num_gpu_blocks,
                    num_cpu_blocks=self.cache_config.num_cpu_blocks,
                    sliding_window=self.cache_config.sliding_window)
        elif CURRENT_VLLM_VERSION in [VLLMVersion.v_0_5_1, VLLMVersion.v_0_6_3]:
            if self.scheduler is None:
                self.scheduler = [
                    Scheduler(self.scheduler_config, self.cache_config, None,
                            self.parallel_config.pipeline_parallel_size)
                    for _ in range(self.parallel_config.pipeline_parallel_size)
                ]

                def get_tokenizer_for_seq(sequence):
                    tokenizer_group = self.get_tokenizer_group()
                    assert tokenizer_group, ("tokenizer_group cannot be None, "
                                            "make sure skip_tokenizer_init is False")
                    return tokenizer_group.get_lora_tokenizer(sequence.lora_request)

                tokenizer_for_seq = get_tokenizer_for_seq if CURRENT_VLLM_VERSION == VLLMVersion.v_0_6_3 \
                    else self.get_tokenizer_for_seq

                self.output_processor = (
                    SequenceGroupOutputProcessor.create_output_processor(
                        self.scheduler_config,
                        self.detokenizer,
                        self.scheduler,
                        self.seq_counter,
                        tokenizer_for_seq,
                        stop_checker=StopChecker(
                            self.scheduler_config.max_model_len,
                            tokenizer_for_seq,
                        ),
                    ))
                if CURRENT_VLLM_VERSION == VLLMVersion.v_0_6_3:
                    self.input_preprocessor = InputPreprocessor(self.model_config,
                                                                self.tokenizer)
                    self.cached_scheduler_outputs = [
                        SchedulerOutputState()
                        for _ in range(self.parallel_config.pipeline_parallel_size)
                    ]
                    self.scheduler_contexts = [
                        SchedulerContext(multi_step_stream_outputs=self.scheduler_config.multi_step_stream_outputs)
                        for _ in range(self.parallel_config.pipeline_parallel_size)
                    ]
                    self.use_cached_outputs = False
                    self.process_request_outputs_callback = None
                    self.tracer = None
            else:
                if CURRENT_VLLM_VERSION == VLLMVersion.v_0_6_3:
                    version = "selfattn"
                    if (self.scheduler_config.embedding_mode
                            or self.cache_config.is_attention_free):
                        version = "placeholder"
                else:
                    version = "v1"
                    if self.scheduler_config.use_v2_block_manager:
                        version = "v2"
                    if self.scheduler_config.embedding_mode:
                        version = "embedding"

                BlockSpaceManagerImpl = get_block_manager_cls(version)
                num_gpu_blocks = self.cache_config.num_gpu_blocks
                if num_gpu_blocks:
                    num_gpu_blocks //= self.pipeline_model_parallel_size()
                num_cpu_blocks = self.cache_config.num_cpu_blocks
                if num_cpu_blocks:
                    num_cpu_blocks //= self.pipeline_model_parallel_size()

                for scheduler in self.scheduler:
                    scheduler.block_manager = BlockSpaceManagerImpl( # pylint: disable=abstract-class-instantiated
                        block_size=self.cache_config.block_size,
                        num_gpu_blocks=num_gpu_blocks,
                        num_cpu_blocks=num_cpu_blocks,
                        sliding_window=self.cache_config.sliding_window,
                        enable_caching=self.cache_config.enable_prefix_caching)

    def _reset_scheduler(self):
        # reset scheduler
        scheduler_list = self.scheduler if isinstance(self.scheduler, list) else [self.scheduler]
        for scheduler in scheduler_list:
            scheduler.block_manager.reset()

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

    def profile_cache_blocks(self):
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        self.clear_cache()

        if CURRENT_VLLM_VERSION == VLLMVersion.v_0_3_0:
            num_gpu_blocks, num_cpu_blocks = self.worker.profile_num_available_blocks(
                self.cache_config.block_size,
                self.cache_config.gpu_memory_utilization,
                self.cache_config.swap_space_bytes,
                self.cache_config.cache_dtype
            )
        elif CURRENT_VLLM_VERSION in [VLLMVersion.v_0_5_1, VLLMVersion.v_0_6_3]:
            num_gpu_blocks, num_cpu_blocks = self.worker.determine_num_available_blocks()
        else:
            raise RuntimeError(f"Unsupported vllm version {CURRENT_VLLM_VERSION}, expect one of {list(VLLMVersion)}")

        self._need_to_reset_scheduler = False
        self.clear_cache()

        return num_gpu_blocks, num_cpu_blocks

    def set_cache_config(self, num_gpu_blocks, num_cpu_blocks):
        # debug log.
        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")

        self._logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                          f"# CPU blocks: {num_cpu_blocks}")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        self._need_to_reset_scheduler = False

    def convert_v1_inputs(self, prompts, prompt_token_ids):
        num_requests = len(prompts)
        assert num_requests == len(prompt_token_ids), \
            ("The lengths of prompts and prompt_token_ids must be the same.")

        inputs = []
        for i in range(num_requests):
            if prompts[i] is None:
                assert isinstance(prompt_token_ids[i], List[int]), \
                    f"Expect prompt_token_ids[{i}] is List[int] when prompt is None, while {prompt_token_ids[i]}."
            if prompt_token_ids[i] is None:
                assert isinstance(prompts[i], str), \
                    f"Expect prompts[{i}] is a string when prompt_token_ids is None, while {prompts[i]}."
            item = TextTokensPrompt(
                prompt=prompts[i],
                prompt_token_ids=prompt_token_ids[i])
            inputs.append(item)

        return inputs

    def _add_request(self, data, is_eval=False): # pylint: disable=arguments-differ
        prompt_key = self.model_args.get("vllm_prompt_key", "prompt")
        input_ids_key = self.model_args.get("vllm_input_ids_key", "input_ids")
        return self._add_request_internal(data[prompt_key], data[input_ids_key], is_eval=is_eval)

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
        min_p = self.model_args.get("eval_min_p", 0.0) if is_eval else self.model_args.get("min_p", 0.0)
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
                    min_p=min_p,
                    use_beam_search=self.model_args.get("use_beam_search"),
                    ignore_eos=self.model_args.get("ignore_eos"),
                    stop=stop,
                    max_tokens=max_tokens,
                    logprobs=1,
                    prompt_logprobs=self.model_args.get("prompt_logprobs", None),
                    skip_special_tokens=self.model_args.get('skip_special_tokens', True)
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
                    min_p=min_p,
                    ignore_eos=self.model_args.get("ignore_eos"),
                    stop=stop,
                    max_tokens=max_tokens,
                    logprobs=1,
                    prompt_logprobs=self.model_args.get("prompt_logprobs", None),
                    skip_special_tokens=self.model_args.get('skip_special_tokens', True)
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
        if self.trainable:
            assert hasattr(self, "model")
            assert hasattr(self, "optimizer")
            assert hasattr(self, "opt_param_scheduler")
            self.model.eval()
        else:
            assert hasattr(self, "model")
            self.model.eval()
        self.worker.model_runner.model = self.model.model
        if CURRENT_VLLM_VERSION == VLLMVersion.v_0_6_3:
            from vllm.worker.multi_step_worker import MultiStepWorker
            if isinstance(self.worker, MultiStepWorker):
                self.worker.model_runner._base_model_runner.model = self.model.model
        if CURRENT_VLLM_VERSION in [VLLMVersion.v_0_5_1, VLLMVersion.v_0_6_3]:
            self.worker.device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self.worker.init_gpu_memory = torch.cuda.mem_get_info()[0]

        if self.module_args.offload_weights:
            if InferenceMemoryManager is None:
                raise Exception("Import InferenceMemoryManager failed, you may need to set right Megatron path first.")
            self._memory_manager = InferenceMemoryManager(
                self.model,
                self.runtime_args.bucket_size_mb_in_memory_manager,
            )
            self.offload()

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

    def tensor_and_expert_model_parallel_size(self):
        """
        get tensor_and_expert_model_parallel_size
        :meta private:
        """
        # vLLM not supported to enable expert parallel size
        # thus: tensor_and_expert_model_parallel_size = tensor_parallel_size
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

    def generate_vllm(self, query, is_eval):
        num_gpu_blocks, num_cpu_blocks = self.profile_cache_blocks()
        num_blocks = torch.tensor([num_gpu_blocks, num_cpu_blocks], device='cuda')
        torch.distributed.all_reduce(num_blocks, op=torch.distributed.ReduceOp.MIN)
        min_gpu_blocks = num_blocks[0].item()
        min_cpu_blocks = num_blocks[1].item()
        self.set_cache_config(min_gpu_blocks, min_cpu_blocks)
        if self.is_last_rank():
            self.build_scheduler()
        self.reinit_cache_engine()
        # add requests of current episode to vllm scheduler
        if self.is_last_rank():
            self._add_request(query, is_eval=is_eval)

        step_outputs = True
        while step_outputs:
            schedule_query = None
            if self.is_last_rank():
                # support multi step schedule.
                virtual_engine = 0
                cached_outputs = self.cached_scheduler_outputs[virtual_engine]
                seq_group_metadata_list = cached_outputs.seq_group_metadata_list
                scheduler_outputs = cached_outputs.scheduler_outputs
                allow_async_output_proc = False

                ctx = self.scheduler_contexts[virtual_engine]

                # Clear outputs for each new scheduler iteration
                ctx.request_outputs.clear()

                # Skip the scheduler if there are any remaining steps in the seq groups.
                # This ensures that the scheduler is only called again when the current
                # batch has completed.
                if not self._has_remaining_steps(seq_group_metadata_list):
                    # Schedule iteration
                    scheduler_outputs = self.schedule()
                    seq_group_metadata_list = scheduler_outputs["seq_group_metadata_list"]

                    ctx.seq_group_metadata_list = seq_group_metadata_list
                    ctx.scheduler_outputs = scheduler_outputs

                    # Maybe switch from async mode to sync mode
                    if not allow_async_output_proc and len(ctx.output_queue) > 0:
                        self._process_model_outputs(ctx=ctx)

                    if (self.scheduler_config.is_multi_step
                            and scheduler_outputs["num_lookahead_slots"] > 0):
                        # cache the scheduler outputs for the next iteration if we have
                        # lookahead slots
                        self._cache_scheduler_outputs_for_multi_step(
                            virtual_engine, seq_group_metadata_list, scheduler_outputs,
                            allow_async_output_proc)

                assert seq_group_metadata_list is not None
                assert scheduler_outputs is not None

                schedule_query = scheduler_outputs
                if len(scheduler_outputs) == 0:
                    if len(ctx.output_queue) > 0:
                        self._process_model_outputs(ctx=ctx)

            schedule_query = broadcast_var_object_dict(schedule_query, torch.distributed.get_world_size()-1)
            output = self.execute_step(schedule_query)
            if self.is_last_rank():
                step_outputs = bool(output)
                signal_tensor = torch.tensor(step_outputs, device='cuda')
                torch.distributed.broadcast(signal_tensor, torch.distributed.get_world_size()-1)
            else:
                signal_tensor = torch.tensor(True, device='cuda')
                torch.distributed.broadcast(signal_tensor, torch.distributed.get_world_size()-1)
            step_outputs = signal_tensor.item()
        if self.is_last_rank():
            self.outputs = sorted(self.outputs, key=lambda x: int(x.request_id))
            return self.outputs

    def schedule(self):
        if self.start_time is None:
            self.start_time = time.monotonic()

        if CURRENT_VLLM_VERSION == VLLMVersion.v_0_6_3:
            for scheduler in self.scheduler:
                self.seq_group_metadata_list, self.scheduler_outputs, _ = scheduler.schedule()
                if self.seq_group_metadata_list:
                    break
        else:
            self.seq_group_metadata_list, self.scheduler_outputs = self.scheduler[0].schedule()

        if self.scheduler_outputs.is_empty():
            return {}

        data = {
            "seq_group_metadata_list" : self.seq_group_metadata_list,
            "blocks_to_swap_in" : self.scheduler_outputs.blocks_to_swap_in,
            "blocks_to_swap_out" : self.scheduler_outputs.blocks_to_swap_out,
            "blocks_to_copy" : self.scheduler_outputs.blocks_to_copy
        }

        if CURRENT_VLLM_VERSION in [VLLMVersion.v_0_5_1, VLLMVersion.v_0_6_3]:
            finished_requests_ids = self.scheduler[0].get_and_reset_finished_requests_ids()
            data.update({
                "num_lookahead_slots": self.scheduler_outputs.num_lookahead_slots,
                "running_queue_size": self.scheduler_outputs.running_queue_size,
                "finished_requests_ids": finished_requests_ids
            })

        return data

    def process_model_outputs(self, output, seq_group_metadata_list=None):
        if CURRENT_VLLM_VERSION == VLLMVersion.v_0_3_0:
            step_outputs = self._process_model_outputs(output, self.scheduler_outputs)
        elif CURRENT_VLLM_VERSION in [VLLMVersion.v_0_5_1, VLLMVersion.v_0_6_3]:
            if CURRENT_VLLM_VERSION == VLLMVersion.v_0_5_1:
                step_outputs = self._process_model_outputs(
                    output, self.scheduler_outputs.scheduled_seq_groups,
                    self.scheduler_outputs.ignored_seq_groups, self.seq_group_metadata_list)
            else:
                # We need to do this here so that last step's sampled_token_ids can
                # be passed to the next iteration for PP.
                # if self.is_last_rank():virtual_engine
                virtual_engine = 0
                allow_async_output_proc = False
                ctx = self.scheduler_contexts[virtual_engine]

                # Clear outputs for each new scheduler iteration
                ctx.request_outputs.clear()
                if self.scheduler_config.is_multi_step:
                    self._update_cached_scheduler_output(virtual_engine, output)
                # Finish the current step for all the sequence groups.
                if self.scheduler_config.is_multi_step:
                    for seq_group in seq_group_metadata_list:
                        seq_group.finish_step()

                if not self._has_remaining_steps(seq_group_metadata_list):
                    # clear the cache if we have finished all the steps.
                    if self.scheduler_config.is_multi_step:
                        self.cached_scheduler_outputs[0] = SchedulerOutputState()

                    # is_first_step_output is True only when the num_steps of all
                    # the sequences are 1. When the num_steps > 1,
                    # multi_step_model_runner does the first-step output append.
                    is_first_step_output: bool = False if not seq_group_metadata_list \
                        else seq_group_metadata_list[0].state.num_steps == 1

                    # Add results to the output_queue
                    ctx.append_output(outputs=output,
                                    seq_group_metadata_list=seq_group_metadata_list,
                                    scheduler_outputs=self.scheduler_outputs,
                                    is_async=allow_async_output_proc,
                                    is_last_step=True,
                                    is_first_step_output=is_first_step_output)

                    self._process_model_outputs(ctx=ctx)
                    if not self.has_unfinished_requests():
                        # Drain async postprocessor (if exists)
                        if len(ctx.output_queue) > 0:
                            self._process_model_outputs(ctx=ctx)
                        assert len(ctx.output_queue) == 0
                    step_outputs = ctx.request_outputs
                else:
                    # Multi-step case
                    step_outputs = ctx.request_outputs
        else:
            raise RuntimeError(f"Unsupported vllm version {CURRENT_VLLM_VERSION}, expect one of {list(VLLMVersion)}")
        done = 0
        for out in step_outputs:
            if out.finished:
                self.outputs.append(out)
                done += 1
                self.pbar.update(1)

        self.num_requests -= done
        if self.num_requests <= 0:
            self.pbar.close()

        if self._log_metrics:
            self.log_metrics_stats(done)

        return self.num_requests

    @torch.inference_mode()
    def execute_step(self, data):
        if CURRENT_VLLM_VERSION == VLLMVersion.v_0_3_0:
            output = self.worker.execute_model(
                data["seq_group_metadata_list"],
                data["blocks_to_swap_in"],
                data["blocks_to_swap_out"],
                data["blocks_to_copy"]
            )
        elif CURRENT_VLLM_VERSION in [VLLMVersion.v_0_5_1, VLLMVersion.v_0_6_3]:
            if CURRENT_VLLM_VERSION == VLLMVersion.v_0_5_1:
                execute_model_req = ExecuteModelRequest(
                    seq_group_metadata_list=data["seq_group_metadata_list"],
                    blocks_to_swap_in=data["blocks_to_swap_in"],
                    blocks_to_swap_out=data["blocks_to_swap_out"],
                    blocks_to_copy=data["blocks_to_copy"],
                    num_lookahead_slots=data["num_lookahead_slots"],
                    running_queue_size=data["running_queue_size"],
                    finished_requests_ids=data["finished_requests_ids"]
                )
                output = self.worker.execute_model(execute_model_req=execute_model_req)
            else:
                if len(data) > 0:
                    # For llm_engine, there is no pipeline parallel support, so the engine
                    # used is always 0.
                    virtual_engine = 0

                    # These are cached outputs from previous iterations. None if on first
                    # iteration
                    seq_group_metadata_list = data["seq_group_metadata_list"]
                    allow_async_output_proc = False

                    assert seq_group_metadata_list is not None
                    finished_requests_ids = data["finished_requests_ids"]

                    # Check if we have a cached last_output from the previous iteration.
                    # For supporting PP this is probably the best way to pass the
                    # sampled_token_ids, as a separate broadcast over all the PP stages
                    # will cause one virtual engine's microbatch to block the pipeline.
                    last_sampled_token_ids = None

                    execute_model_req = ExecuteModelRequest(
                        seq_group_metadata_list=seq_group_metadata_list,
                        blocks_to_swap_in=data["blocks_to_swap_in"],
                        blocks_to_swap_out=data["blocks_to_swap_out"],
                        blocks_to_copy=data["blocks_to_copy"],
                        num_lookahead_slots=data["num_lookahead_slots"],
                        running_queue_size=data["running_queue_size"],
                        finished_requests_ids=finished_requests_ids,
                        # We use ExecuteModelRequest to pass the last sampled_token_ids
                        # to each of the non-last PP stages for in-place prepare_input.
                        last_sampled_token_ids=last_sampled_token_ids)

                    if allow_async_output_proc:
                        execute_model_req.async_callback = self.async_callbacks[
                            virtual_engine]
                    output = self.worker.execute_model(execute_model_req=execute_model_req)
                else:
                    # No outputs in this case
                    output = []
        else:
            raise RuntimeError(f"Unsupported vllm version {CURRENT_VLLM_VERSION}, expect one of {list(VLLMVersion)}")

        if self.is_last_rank() and hasattr(self, "scheduler_outputs"):
            return self.process_model_outputs(output, seq_group_metadata_list=data["seq_group_metadata_list"])

        return output

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

    def set_vllm_pp_layer_partition(self):
        pipeline_world_size = self.module_args.pipeline_model_parallel_size
        num_layers = self.model_args.get("num_layers")
        remainder = num_layers % pipeline_world_size
        if not self.model_args.get("allow_padding_num_layers", None):
            assert remainder == 0, \
                f"expect num_layers % pipeline_model_size == 0 when VLLM_PP_LAYER_PARTITION is not set. \
                while num_layers = {num_layers} pipeline_model_size = {pipeline_world_size}"
            return

        if remainder > 0:
            assert not self.model_args.get("standalone_embedding_stage", False), \
                "not support standalone embedding stage if allow_padding_num_layers is true"
            # pad num_layers to make num_layers % pipeline_model_parallel_size == 0
            num_layers_with_padding = num_layers - remainder + pipeline_world_size
        else:
            num_layers_with_padding = num_layers
        num_layers_without_padding = num_layers
        num_layers = num_layers_with_padding
        num_layers_per_stage_with_padding = (
            num_layers // pipeline_world_size)

        # Each stage gets a contiguous set of layers.
        if self.model_args.get("pipeline_layers", None) is not None:
            rank_sizes = self.model_args.get("pipeline_layers", None)
            assert isinstance(rank_sizes, list) and all(isinstance(ele, int) for ele in rank_sizes), \
                f"pipeline_layers expected to be list, and num layer of each stage to be integer, while {rank_sizes}."
        else:
            rank_sizes = [num_layers_per_stage_with_padding] * pipeline_world_size
            num_padding = num_layers - num_layers_without_padding
            if num_padding > 0:
                assert num_padding == 2, \
                    "Support num_padding_lsyers == 2 when applies inbalanced pp. Please set `args.pipeline_layers` for VLLMModule."

            for _index in range(-1, num_padding - 1):
                rank_sizes[_index] -= 1
        assert len(rank_sizes) == pipeline_world_size

        # set env variable VLLM_PP_LAYER_PARTITION
        vllm_pp_layer_partition = ",".join([str(ele) for ele in rank_sizes])
        if os.getenv("VLLM_PP_LAYER_PARTITION", None) is not None:
            env_vllm_pp_layer_partition = os.getenv("VLLM_PP_LAYER_PARTITION", None)
            if vllm_pp_layer_partition != env_vllm_pp_layer_partition:
                self._logger.warning(
                    f"expect VLLM_PP_LAYER_PARTITION to be {vllm_pp_layer_partition}, while {env_vllm_pp_layer_partition}")
        os.environ["VLLM_PP_LAYER_PARTITION"] = vllm_pp_layer_partition

    def log_metrics_stats(self, num_done_requests):
        now = time.monotonic()
        self.num_done_requests += num_done_requests
        scheduler_list = self.scheduler if isinstance(self.scheduler, list) else [self.scheduler]
        avg_request_throughput = self.num_done_requests / (now - self.start_time)
        if self.scheduler_outputs.prompt_run:
            self.num_processed_prompt += self.scheduler_outputs.num_batched_tokens
        else:
            self.num_generated_tokens += self.scheduler_outputs.num_batched_tokens

        avg_generation_throughput = self.num_generated_tokens / (now - self.start_time)
        avg_prompt_throughput = self.num_processed_prompt / (now - self.start_time)

        self.forward_count += 1
        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = sum(
            scheduler.block_manager.get_num_free_gpu_blocks() for scheduler in scheduler_list)
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        self.gpu_cache_usage += num_used_gpu_blocks / total_num_gpu_blocks
        avg_gpu_cache_usage = self.gpu_cache_usage / self.forward_count

        total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = sum(
                scheduler.block_manager.get_num_free_cpu_blocks() for scheduler in scheduler_list)
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0

        self.cpu_cache_usage += cpu_cache_usage
        avg_cpu_cache_usage = self.cpu_cache_usage / self.forward_count

        for idx in range(self.num_done_requests - num_done_requests, self.num_done_requests):
            output = self.outputs[idx]
            prompt_length = len(output.prompt_token_ids)
            output_length = len(output.outputs[0].token_ids)
            batch_index = int(output.request_id / self.scheduler_config.max_num_seqs)
            self.max_prompt_length_static_batching[batch_index] = max(
                self.max_prompt_length_static_batching[batch_index], prompt_length)
            self.max_output_length_static_batching[batch_index] = max(
                self.max_output_length_static_batching[batch_index], output_length)
            self.action_length += output_length
            self.action_max_length = max(self.action_max_length, output_length)
            self.action_min_length = min(self.action_min_length, output_length)
        action_length_mean = float(self.action_length / self.num_done_requests) if self.num_done_requests else 0.0

        for scheduler in scheduler_list:
            self.batch_size_stats += len(scheduler.running)
        avg_batch_size = self.batch_size_stats / self.forward_count

        if not self.num_requests or (now - self.last_stats_time >= _LOGGING_INTERVAL_SEC):
            self.last_stats_time = now
            message = ""
            if not self.num_requests:
                batch_size = [self.scheduler_config.max_num_seqs \
                    for _ in range(math.ceil(self.num_done_requests / self.scheduler_config.max_num_seqs))]
                if self.num_done_requests % self.scheduler_config.max_num_seqs:
                    batch_size[-1] = self.num_done_requests % self.scheduler_config.max_num_seqs
                num_prompt_tokens_static_batching = sum( # pylint: disable=consider-using-generator
                    [prompt_len * bs for prompt_len, bs in zip(self.max_prompt_length_static_batching, batch_size)])
                num_output_tokens_static_batching = sum( # pylint: disable=consider-using-generator
                    [output_length * bs for output_length, bs in zip(self.max_output_length_static_batching, batch_size)])
                message = f"num_processed_prompts_continuous_batching: {self.num_processed_prompt}, " \
                          f"num_processed_prompts_static_batching: {num_prompt_tokens_static_batching}, " \
                          f"num_processed_prompts_continuous_batching/num_processed_prompts_static_batching: \
                          {self.num_processed_prompt/num_prompt_tokens_static_batching:.1f}, " \
                          f"num_output_tokens_continuous_batching: {self.num_generated_tokens}, " \
                          f"num_output_tokens_static_batching: {num_output_tokens_static_batching}, " \
                          f"num_output_tokens_continuous_batching/num_output_tokens_static_batching: \
                          {self.num_generated_tokens/num_output_tokens_static_batching:.1f}, " \

            self._logger.info(f"allready generate responses for {self.num_done_requests} reqs, "
                              f"avg_request_throughput: {avg_request_throughput:.1f} reqs/s, "
                              f"avg_prompt_throughput: {avg_prompt_throughput:.1f} tokens/s, "
                              f"avg_generation_throughput: {avg_generation_throughput:.1f} tokens/s, "
                              f"avg_batch_size: {avg_batch_size:.1f} reqs, "
                              f"avg_gpu_cache_usage: {avg_gpu_cache_usage * 100:.1f}%, "
                              f"avg_cpu_cache_usage {avg_cpu_cache_usage * 100:.1f}%, "
                              f"action_length_mean: {action_length_mean:.1f}, "
                              f"action_max_length: {self.action_max_length if self.num_done_requests else 'inf'}, "
                              f"action_min_length: {self.action_min_length if self.num_done_requests else '-inf'}, "
                              f"{message}")
# pylint: enable=import-outside-toplevel,unexpected-keyword-arg,no-value-for-parameter,too-many-function-args
