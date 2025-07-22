# pylint: skip-file
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
import inspect
import os
from contextlib import nullcontext
from typing import Union
from collections import defaultdict

import torch
from megatron.core import mpu
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec, get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec, get_gpt_mtp_block_spec)
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import get_model_config
from megatron.training import get_args, get_model, get_timers, print_rank_0, get_tokenizer
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.training import setup_model_and_optimizer
from megatron.training.utils import (calc_params_l2_norm,
                                     logical_and_across_model_parallel_group)
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

import chatlearn
from chatlearn import MegatronModule
from chatlearn.algorithm.grpo_utils.megatron_utils import (
    PolicyModel, forward_step, inference_forward_step,
    training_log)

REF_TAG = "ref_logprobs"
OLD_TAG = "old_logprobs"


class MegatronPolicyTrainer(MegatronModule):
    """MegatronPolicyTrainer"""

    def setup(self):
        self.stats = {}
        self.buffer = {}
        self.args = get_args()
        self.report_memory_flag = True
        self.iteration_for_log = 0
        if getattr(self.global_args, "padded_vocab_size", None) is None:
            get_args().padded_vocab_size = self.args.vocab_size

        if self.trainable:
            self.model, self.optimizer, self.opt_param_scheduler = (
                setup_model_and_optimizer(
                    self.model_provider, ModelType.encoder_or_decoder
                )
            )
            self.config = get_model_config(self.model[0])
            self.config.grad_scale_func = self.optimizer.scale_loss
            self.config.finalize_model_grads_func = finalize_model_grads

        else:
            self.model = get_model(self.model_provider, wrap_with_ddp=False)
            if self.args.load is not None:
                print(f"reference loading : {self.args.load}")
                _, _ = load_checkpoint(
                    self.model,
                    None,
                    None,
                    checkpointing_context={},
                    skip_load_to_model_and_opt=False
                    and getattr(self.args, "use_torch_fsdp2", False)
                    and self.args.ckpt_format == "torch_dist",
                )
            if int(os.environ.get("WORLD_SIZE", 1)) > 1:
                torch.distributed.barrier(
                    device_ids=[int(os.environ.get("LOCAL_RANK", 0))]
                )

    def model_provider(self, pre_process=True, post_process=True) -> Union[PolicyModel]:
        args = get_args()
        use_te = args.transformer_impl == "transformer_engine"

        if args.record_memory_history:
            torch.cuda.memory._record_memory_history(
                True,
                # keep 100,000 alloc/free events from before the snapshot
                trace_alloc_max_entries=100000,
                # record stack information for the trace events
                trace_alloc_record_context=True,
            )

            def oom_observer(device, alloc, device_alloc, device_free):
                # snapshot right after an OOM happened
                print("saving allocated state during OOM")
                snapshot = torch.cuda.memory._snapshot()
                from pickle import dump

                dump(
                    snapshot,
                    open(
                        f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}",
                        "wb",
                    ),
                )

            torch._C._cuda_attach_out_of_memory_observer(oom_observer)

        print_rank_0("building GPT model ...")
        # Experimental loading arguments from yaml
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config,
                    use_transformer_engine=use_te,
                    normalization=args.normalization,
                )
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.multi_latent_attention,
                        args.moe_use_legacy_grouped_gemm,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.multi_latent_attention,
                        args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization,
                    )
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec, use_transformer_engine=use_te
            )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if (
                    "preserve_high_precision_init_val"
                    in inspect.signature(fp8_model_init).parameters
                ):
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                )

        with build_model_context(**build_model_context_args):
            model = PolicyModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling,
                mtp_block_spec=mtp_block_spec,
                module_args=self.module_args
            )

        return model

    def train_step(self, data_list):
        args = get_args()
        timers = get_timers()
        self.model[0].module.train()
        data_iterator = iter(data_list)

        self.optimizer.zero_grad()

        # Forward pass.
        timers("forward-backward", log_level=1).start(barrier=args.barrier_with_L1_time)
        forward_backward_func = get_forward_backward_func()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
        )

        timers("forward-backward").stop()

        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Update parameters.
        timers("optimizer", log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        timers("optimizer").stop()

        update_successful = logical_and_across_model_parallel_group(update_successful)

        # Update learning rate.
        if update_successful:
            increment = (
                get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
            )
            self.opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        # NOTE: per-token-average metrics, besides loss_per_microbatch,
        # losses_reduced also contains num_tokens_per_microbatch
        loss_reduced_for_metric = {}
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            total_losses = defaultdict(list)
            for losses_per_micro_batch in losses_reduced:
                for k, v in losses_per_micro_batch.items():
                    total_losses[k].append(v)
            # sum across microbatches.
            keys = sorted(list(total_losses.keys()))
            keys.pop(keys.index('num_tokens'))
            loss_for_dp_reduce = torch.stack(
                [sum(total_losses[key]).float() for key in keys] +
                [sum(total_losses.pop('num_tokens')).float()]
            )
            torch.distributed.all_reduce(loss_for_dp_reduce, group=mpu.get_data_parallel_group())
            loss_reduced_for_metric = {
                key: (loss_for_dp_reduce[i] / loss_for_dp_reduce[-1]).cpu().item()
                for i, key in enumerate(keys)
            }
            
        self.iteration_for_log += 1

        self.args.consumed_train_samples += (
            mpu.get_data_parallel_world_size()
            * self.args.micro_batch_size
            * get_num_microbatches()
        )

        # Logging.
        loss_scale = self.optimizer.get_loss_scale().item()
        params_norm = None
        if self.args.log_params_norm:
            params_norm = calc_params_l2_norm(self.model)
        report_memory_flag = training_log(
            loss_reduced_for_metric,
            {},
            self.optimizer.param_groups[0]["lr"],
            self.iteration_for_log,
            loss_scale,
            self.report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
            self.stats,
            {},
            "policy_trainer",
            self._metric_list,
        )

        self.report_memory_flag = report_memory_flag

    @torch.no_grad()
    def forward_step(self, data):
        """Do an inference forward step. Only for computation of old_logprobs and ref_logprobs.

        Args:
            data (Dict[str, Sequence[Any]]): The batched data with shape of [generation_batch_size, *].

        Returns:
            data (Dict[str, Any]): If this is the last rank of the replica, the output logprobs will be
             updated into the dict, otherwise do NOTHING.
        """
        for model_module in self.model:
            model_module.eval()
        num_microbatches, data_iter = self._chunk_global_batch_into_micro_batch(data)

        # NOTE: internal computation
        args = get_args()
        forward_backward_func = get_forward_backward_func()
        forward_data_store: List[Any] = forward_backward_func(
            forward_step_func=inference_forward_step,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=True,
            collect_non_loss_data=True,  # set True to hack the forward_step_func
        )  # shape: [num_microbatches, *]

        for model_module in self.model:
            model_module.train()
        if not mpu.is_pipeline_last_stage():
            return
        # trainable is True --> policy trainer; False --> PolicyReference
        tag = OLD_TAG if self.trainable else REF_TAG
        logprobs = -torch.cat(forward_data_store, dim=0)
        assert (
            tag not in data
        ), f"The {tag} is already computed in this batch, old_value: {data[tag]}, new_value: {logprobs}"
        data.update({tag: logprobs})
        return data

    def _chunk_global_batch_into_micro_batch(self, data_dict):
        # NOTE: assume that generation_batch_size is multiple of micro_batch_size in MCore
        assert len(data_dict) > 0, "Expect non-empty data_dict"
        args = get_args()
        generation_batch_size = self.module_args.generation_batch_size
        micro_batch_size = args.micro_batch_size
        num_microbatches = generation_batch_size // micro_batch_size

        def _data_iter(data_dict):
            for i in range(num_microbatches):
                yield {
                    k: v[i * micro_batch_size : (i + 1) * micro_batch_size]
                    for k, v in data_dict.items()
                }

        return num_microbatches, _data_iter(data_dict)
