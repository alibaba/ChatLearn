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
"""Training utilities."""

import dataclasses
import torch
from megatron.training import get_args
from megatron.training import get_timers
from megatron.training import get_tokenizer
try:
    from megatron.training import get_num_microbatches
except ImportError:
    from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import print_rank_0
from megatron.core.enums import ModelType
try:
    from megatron.core.utils import get_model_config
except ImportError:
    get_model_config = None
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.training import get_model
from megatron.training.training import get_optimizer_param_scheduler
from megatron.training.utils import unwrap_model
from megatron.training.training import train_step as megatron_train_step
from megatron.core import mpu
from megatron.core.pipeline_parallel import get_forward_backward_func

from chatlearn import MegatronModule
from chatlearn.utils.megatron_utils import load_checkpoint
from .constants import TrainerEngine


class BaseTrainer(MegatronModule):
    """Base Trainer"""

    def setup(self):
        self.buffer = {}
        self.stats = {}

        self.args = get_args()

        print(f"{self.name} trainer loading : {self.args.load}")

        self.model_type = ModelType.encoder_or_decoder
        self.tokenizer = get_tokenizer()

        self.args.save = f"{self.runtime_args.output_dir}/save_model/{self.name}/"

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
        if args.load and (args.finetune or args.no_load_optim):
            torch.distributed.barrier()
            args.iteration = load_checkpoint(model, None, None, strict=strict,
                adaptive_parallel_strategy=self.args.adaptive_parallel_strategy_on_checkpoint)
            torch.distributed.barrier()

        unwrapped_model = unwrap_model(model)
        kwargs = {}
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)
        config.timers = get_timers()

        if args.load and args.no_load_optim:
            optimizer = None
            opt_param_scheduler = None
        else:
            optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
            opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

        if args.load and not args.finetune and not args.no_load_optim:
            timers = get_timers()
            timers('load-checkpoint', log_level=0).start(barrier=True)
            args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler, strict=strict)
            timers('load-checkpoint').stop(barrier=True)
            if args.no_load_optim:
                optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
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
            optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
            opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

        return model, optimizer, opt_param_scheduler

    def dpo_train_step(self, forward_step_func, data_iterator,
                       model, optimizer, opt_param_scheduler, config):
        # Code below is migrated from 'train_step' function of Megatron-LM:2ca5cb09
        args = get_args()
        timers = get_timers()

        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        micro_batch_size = args.micro_batch_size
        if config.pipeline_model_parallel_size > 1:
            micro_batch_size *= 2

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)

        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Vision gradients.
        if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(model[0])
            unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

        # Update parameters.
        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        timers('optimizer').stop()

        # Vision momentum.
        if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(model[0])
            unwrapped_model.update_momentum(args.curr_iteration)

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0].keys():
                numerator = 0
                denominator = 0
                for x in losses_reduced:
                    val = x[key]
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    if isinstance(val, tuple) or isinstance(val, list): # pylint: disable=consider-merging-isinstance
                        numerator += val[0]
                        denominator += val[1]
                    else:
                        # legacy behavior. we average over the number of microbatches,
                        # and so the denominator is 1.
                        numerator += val
                        denominator += 1
                loss_reduced[key] = numerator / denominator
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        return {}, skipped_iter, grad_norm, num_zeros_in_grad


    def train_step(self, data, iteration=None):
        assert isinstance(data, list)
        data_iterator = iter(data)
        if self.config is not None:
            kwargs = {"config": self.config}
        else:
            kwargs = {}
        if self.args.trainer_engine == TrainerEngine.DPO:
            _, skipped_iter, grad_norm, num_zeros_in_grad = self.dpo_train_step(self._forward_step, data_iterator,
                                                                                self.model, self.optimizer,
                                                                                self.opt_param_scheduler, **kwargs)
        else:
            _, skipped_iter, grad_norm, num_zeros_in_grad = megatron_train_step(self._forward_step, data_iterator,
                                                                                self.model, self.optimizer,
                                                                                self.opt_param_scheduler, **kwargs)
        self.post_update_stuffs({}, skipped_iter,
                                grad_norm, num_zeros_in_grad, iteration)
