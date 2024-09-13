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
"""DeepSpeed module"""

from datetime import timedelta
import math
import os
import random
import deepspeed
import numpy as np
import torch
from torch import distributed as dist
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.integrations import HfDeepSpeedConfig
from transformers.trainer import get_scheduler

from chatlearn.utils.utils import dict_to_simplenamespace
from .deepspeed.deepspeed_utils import get_eval_ds_config, get_tokenizer, get_train_ds_config, create_optimizer
from .deepspeed.deepspeed_utils import save_hf_format, save_zero_three_model
from .torch_module import TorchModule


class DeepSpeedModule(TorchModule):
    """DeepSpeedModule is the class for models accelerated with DeepSpeed.

    Args
    ----
    name : str
        model name
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.trainable:
            # inference only
            if self.model_args.get("train_micro_batch_size") != self.module_args.generation_batch_size:
                self._logger.info(
                    f"{self.name} Overwrite train_micro_batch_size with generation_batch_size {self.module_args.generation_batch_size}")
            self.train_micro_batch_size = self.module_args.generation_batch_size
        else:
            self.train_micro_batch_size = self.runtime_args.train_micro_batch_size
            self.train_global_batch_size = self.runtime_args.train_global_batch_size

        self.zero_size = self.module_args.zero_size

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout):
        self.set_seed(self.seed)
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed(timeout=timeout)

    def prepare(self, *models_or_model_optim_pairs):
        ret = []
        for arg in models_or_model_optim_pairs:
            if not isinstance(arg, tuple):
                ret.append(self._ds_init_eval_model(arg))
            else:
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._ds_init_train_model(*arg))

        return ret[0] if len(ret) == 1 else ret

    def _ds_init_eval_model(self, model):
        ds_config = self.get_ds_eval_config(offload=getattr(model, "_offload", False))
        local_rank = int(os.environ['LOCAL_RANK'])

        engine, *_ = deepspeed.initialize(
            model=model,
            args={"local_rank": local_rank},
            config=ds_config,
            dist_init_required=True,
        )
        model = engine
        return model

    def _ds_init_train_model(self, model, optim, scheduler):
        ds_config = self.get_ds_train_config()
        local_rank = int(os.environ['LOCAL_RANK'])

        engine, optim, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optim,
            lr_scheduler=scheduler,
            config=ds_config,
            args={"local_rank": local_rank},
            dist_init_required=True,
        )
        model = engine
        return model, optim, scheduler

    def get_ds_eval_config(self, offload=False):
        # DS Config
        ds_config = get_eval_ds_config(offload=offload, stage=self.zero_stage if self.zero_stage == 3 else 0, bf16=self.bf16)
        ds_config["train_micro_batch_size_per_gpu"] = self.train_micro_batch_size
        ds_config["train_batch_size"] = self.train_micro_batch_size * self.zero_size
        return ds_config

    def get_ds_train_config(self):
        # DS Config
        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=self.adam_offload,
            stage=self.zero_stage,
            bf16=self.bf16,
            max_norm=self.max_norm,
            grad_accum_dtype="fp32",
            disable_trace_cache=self.disable_trace_cache,
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.train_micro_batch_size
        ds_config["gradient_accumulation_steps"] = self.train_global_batch_size // self.train_micro_batch_size // self.world_size
        ds_config["train_batch_size"] = self.train_global_batch_size

        return ds_config

    def create_model(self, args):
        # TODO: try attn_implementation="flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrain_or_model,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if self.bf16 else "auto"
        )
        return model


    def model_setup(self):
        super().model_setup()
        args = dict_to_simplenamespace(self.model_args)
        self.prompt_max_len = getattr(args, "prompt_max_len", 1024)
        self.args = args
        self.zero_stage = getattr(args, "zero_stage", 3)
        self.bf16 = args.bf16
        self.seed = getattr(args, "seed", 42)
        self.max_norm = getattr(args, "max_norm", 1.0)
        dist_timeout = getattr(args, 'distributed_timeout', 30)
        self.setup_distributed(timedelta(minutes=dist_timeout))
        # TODO: deal with offload later
        ds_config = self.get_ds_eval_config(offload=False)
        # efficiently deploy DeepSpeed stage 3, you must instantiate the HfDeepSpeedConfig
        # object before instantiating the model.
        # https://huggingface.co/transformers/v4.9.2/main_classes/deepspeed.html
        dschf = HfDeepSpeedConfig(ds_config) if ds_config is not None and self.zero_stage == 3 else None # pylint: disable=unused-variable
        model = self.create_model(self.args)
        self.tokenizer = get_tokenizer(
            args.pretrain_or_model, model, "left", use_fast=True
        )
        if self.trainable:
            if getattr(args, "gradient_checkpointing", False):
                model.gradient_checkpointing_enable()
            self.disable_trace_cache = True
            learning_rate = float(args.learning_rate)
            self.adam_offload = False
            num_update_steps_per_episodes = self.runtime_args.sample_per_episode // self.train_global_batch_size
            l2 = float(args.l2)
            max_steps = math.ceil(self.runtime_args.num_episode * num_update_steps_per_episodes)
            optimizer = create_optimizer(
                model, self.adam_offload, lr=learning_rate, betas=(0.9, 0.95), weight_decay=l2
            )
            scheduler = get_scheduler("cosine_with_min_lr",
                optimizer,
                num_warmup_steps=math.ceil(max_steps * 0.03),
                num_training_steps=max_steps,
                scheduler_specific_kwargs={"min_lr": learning_rate * 0.1},)
            self.model, self.optimizer, self.scheduler = self.prepare((model, optimizer, scheduler))
        else:
            self.model = self.prepare(model)
        self.generation_config = GenerationConfig.from_pretrained(args.pretrain_or_model, trust_remote_code=True)
        self.tokenizer.eos_token_id = self.generation_config.eos_token_id

        if not self.trainable:
            self.model.eval()

    @property
    def data_parallel_size(self):
        """
        :meta private:
        """
        return dist.get_world_size()

    @property
    def data_parallel_rank(self):
        """
        :meta private:
        """
        return dist.get_rank()

    def save_checkpoint(self, iteration):
        save_dir = f"{self.runtime_args.output_dir}/save_model/{self.name}/{iteration}"
        save_hf_format(self.model, self.tokenizer, save_dir)
        save_zero_three_model(self.model, torch.distributed.get_rank(), save_dir, self.zero_stage)
        self._logger.info(f"save checkpoint to {save_dir}")
