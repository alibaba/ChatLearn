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
"""Training utilities."""

import torch
from megatron import get_args
from megatron import get_timers
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.core.enums import ModelType
try:
    from megatron.core.utils import get_model_config
except ImportError:
    get_model_config = None
from megatron.optimizer import get_megatron_optimizer
from megatron.training import get_optimizer_param_scheduler, get_model
from megatron.utils import unwrap_model
from megatron.training import train_step as megatron_train_step

from chatlearn import RLHFMegatronModule
from chatlearn.utils.megatron_utils import load_checkpoint


class BaseTrainer(RLHFMegatronModule):
    """Base Trainer"""

    def setup(self):
        self.buffer = {}
        self.stats = {}

        self.args = get_args()

        print(f"value trainer loading : {self.args.load}")

        self.model_type = ModelType.encoder_or_decoder
        self.tokenizer = get_tokenizer()

        self.args.save = f"{self.args.save}/{self.name}/"

        if self.resume_training:
            self.args.load = get_args().save
            self.args.load_iteration = -1  # latest
            self.args.no_load_optim = False  # latest
            self.args.no_load_rng = False  # latest
            self.args.no_load_args = False  # latest
            self.args.no_load_scheduler = False  # latest
            self.args.adaptive_parallel_strategy_on_checkpoint = False
            self.args.finetune = False
            print(
                f"{self.name} continue train args load: {self.args.load} self.args.load_iteration {self.args.load_iteration}")

        self.model, self.optimizer, self.opt_param_scheduler = self.setup_model_and_optimizer(self.model_provider,
                                                                                              self.model_type)
        if get_model_config is not None:
            self.config = get_model_config(self.model[0])
            self.config.grad_scale_func = self.optimizer.scale_loss
        else:
            self.config = None


    def setup_model_and_optimizer(self, model_provider_func,
                                  model_type,
                                  no_wd_decay_cond=None,
                                  scale_lr_cond=None,
                                  lr_mult=1.0):
        """Setup model and optimizer."""
        args = get_args()
        model = get_model(model_provider_func, model_type)
        self.model = model

        if self.enable_lora:
            # ignore layer related params
            strict = False
        else:
            strict = True
        if args.load is not None and (args.finetune or args.no_load_optim):
            torch.distributed.barrier()
            args.iteration = load_checkpoint(model, None, None, strict=strict,
                adaptive_parallel_strategy=self.args.adaptive_parallel_strategy_on_checkpoint)
            torch.distributed.barrier()

        unwrapped_model = unwrap_model(model)

        if args.load and args.no_load_optim:
            optimizer = None
            opt_param_scheduler = None
        else:
            optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
            opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

        if args.load is not None and not args.finetune and not args.no_load_optim:
            timers = get_timers()
            timers('load-checkpoint', log_level=0).start(barrier=True)
            args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler, strict=strict)
            timers('load-checkpoint').stop(barrier=True)
            if args.no_load_optim:
                optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                                   scale_lr_cond, lr_mult)
                opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
                opt_param_scheduler.step(args.iteration * args.global_batch_size)
            timers.log(['load-checkpoint'])
        else:
            args.iteration = 0

        # get model without FP16 and/or TorchDDP wrappers
        if args.iteration == 0 and len(unwrapped_model) == 1 \
            and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
            print_rank_0("Initializing ICT from pretrained BERT model")
            unwrapped_model[0].init_state_dict_from_bert()
            if args.fp16:
                optimizer.reload_model_params()
        if optimizer is None:
            optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
            opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

        return model, optimizer, opt_param_scheduler

    def train_step(self, data, train_info):
        iteration = train_info["iteration"]
        data_iterator = iter(data)
        if self.config is not None:
            kwargs = {"config": self.config}
        else:
            kwargs = {}
        _, skipped_iter, grad_norm, num_zeros_in_grad = megatron_train_step(self._forward_step, data_iterator,
                                                                            self.model, self.optimizer,
                                                                            self.opt_param_scheduler, **kwargs)
        self.post_update_stuffs({}, skipped_iter,
                                grad_norm, num_zeros_in_grad, iteration)
