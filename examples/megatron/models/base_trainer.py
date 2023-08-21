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
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.optimizer import get_megatron_optimizer
from megatron.training import get_optimizer_param_scheduler, get_model
from megatron.utils import unwrap_model
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from chatlearn import RLHFMegatronModule


class BaseTrainer(RLHFMegatronModule):
    """Base Trainer"""

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
            args.iteration = load_checkpoint(model, None, None, \
                                             adaptive_parallel_strategy=args.adaptive_parallel_strategy_on_checkpoint,
                                             strict=strict)
            torch.distributed.barrier()

        unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))

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
            args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler, \
                                             adaptive_parallel_strategy=args.adaptive_parallel_strategy_on_checkpoint,
                                             strict=strict)
            timers('load-checkpoint').stop(barrier=True)
            if args.no_load_optim:
                optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                                   scale_lr_cond, lr_mult)
                opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
                opt_param_scheduler.step(args.iteration * args.global_batch_size)
            timers.log(['load-checkpoint'])
        else:
            args.iteration = 0

        # We only support local DDP with multiple micro-batches.
        if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
            assert args.DDP_impl == 'local'

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
