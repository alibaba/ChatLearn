# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#chatlearn/synchronizer/__init__.py
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Megatron module"""
import re
import torch
import torch.distributed as dist

try:
    from megatron.training import get_args
    from megatron.training.arguments import parse_args
    from megatron.core import parallel_state as mpu
    from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
    from megatron.training.training import save_checkpoint_and_time
    IS_MEGATRON_SUPPORTED = True
except ImportError:
    IS_MEGATRON_SUPPORTED = False

from .torch_module import TorchModule

if IS_MEGATRON_SUPPORTED:
    try:
        # pylint: disable-next=import-outside-toplevel, unused-import
        from megatron.core.distributed.distributed_data_parallel import _ParamAndGradBuffer
    except ImportError as exc:
        raise ValueError(
            'Old or customed version of Megatron is no longer supported. Please checkout to 0f4e0e1872b62a96d0465de477f26ae81a2e33d7'
        ) from exc

    from chatlearn.models.megatron.memory_manager import InferenceMemoryManager, TrainerMemoryManager

    class MegatronModule(TorchModule):
        """MegatronModule is the class for Alignment Megatron models.

        Args
        ----
        name : str
            model name
        """

        def add_extra_args(self, parser):
            """
            Add extra arguments for megatron.

            Args
            ----
            parser : ArgumentParser
                Add extra arguments.
            """
            return parser

        def init(self):
            """
            :meta private:
            """
            args = parse_args(self.add_extra_args, ignore_unknown_args=True)

            def try_convert_to_default_type(default_value, value):
                """Convert value to type(default_value) if possible"""
                # NOTE: For complex cases, e.g., moe_layer_freq, default_type may differ from value_type
                if default_value is None or value is None:
                    return value
                default_type = type(default_value)
                if not isinstance(value, default_type):
                    try:
                        return default_type(value)
                    except Exception:
                        pass
                return value

            if self.module_args is not None:
                for key, value in self.module_args.items():
                    setattr(args, key, try_convert_to_default_type(getattr(args, key, None), value))
            # settings for mcore parameters micro_batch_size and global_batch_size by chatlearn args
            args.micro_batch_size = self.runtime_args.train_micro_batch_size if self.trainable else self.module_args.generation_batch_size
            args.global_batch_size = self.runtime_args.train_global_batch_size if self.trainable else self.module_args.generation_batch_size

            initialize_megatron(parsed_args=args)

            if self.trainable:
                # slow down if set jit fusion for inference model
                set_jit_fusion_options()

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
                if self.module_args.free_gpu_memory.offload_weights or \
                    self.module_args.free_gpu_memory.free_grad_buffers or \
                    self.module_args.free_gpu_memory.offload_optimizer_states:
                    self._memory_manager = TrainerMemoryManager(
                        self.megatron_model(),
                        self.optimizer,
                        self.megatron_args.use_distributed_optimizer,
                        self.megatron_args.accumulate_allreduce_grads_in_fp32,
                        self.megatron_args.params_dtype,
                        self.runtime_args.bucket_size_mb_in_memory_manager,
                    )
                    self.offload()
            else:
                assert hasattr(self, "model")
                self.megatron_model().eval()
                if self.module_args.free_gpu_memory.offload_weights:
                    self._memory_manager = InferenceMemoryManager(
                        self.megatron_model(),
                        self.runtime_args.bucket_size_mb_in_memory_manager,
                    )
                    self.offload()
            self.set_pipe_layer_num_offset()

        def set_pipe_layer_num_offset(self):
            self.stage2layer_num = [None] * self.pipeline_model_parallel_size()
            self.stage2offset = [0] * self.pipeline_model_parallel_size()
            stage_layer_num = self.get_pipeline_stage_layer_num()
            world_size = torch.distributed.get_world_size()
            rank_layer_num = torch.tensor([self.pipeline_parallel_rank(), stage_layer_num], device='cuda')
            # Gather all tensors to all processes
            all_stage_layer_nums = [torch.zeros_like(rank_layer_num, device='cuda') for _ in range(world_size)]
            torch.distributed.all_gather(all_stage_layer_nums, rank_layer_num)
            for item in all_stage_layer_nums:
                rank = item[0].item()
                num = item[1].item()
                if self.stage2layer_num[rank] is None:
                    self.stage2layer_num[rank] = num
                else:
                    assert self.stage2layer_num[rank] == num
            for i, num in enumerate(self.stage2layer_num):
                if i+1 == len(self.stage2offset):
                    break
                self.stage2offset[i+1] = self.stage2offset[i] + num

        @property
        def megatron_args(self):
            """
            :meta private:
            """
            return get_args()

        def pipeline_model_parallel_size(self):
            """
            get pipeline_model_parallel_size

            :meta private:
            """
            return self.megatron_args.pipeline_model_parallel_size

        def tensor_model_parallel_size(self):
            """
            get tensor_model_parallel_size

            :meta private:
            """
            return self.megatron_args.tensor_model_parallel_size

        def expert_model_parallel_size(self):
            """
            get expert_model_parallel_size
            :meta private:
            """
            if hasattr(self.megatron_args, "expert_model_parallel_size"):
                return self.megatron_args.expert_model_parallel_size
            if hasattr(self.megatron_args, "moe_expert_model_parallel_size"):
                return self.megatron_args.moe_expert_model_parallel_size
            return 1

        def tensor_and_expert_model_parallel_size(self):
            """
            get tensor_and_expert_model_parallel_size
            :meta private:
            """
            return self.megatron_args.tensor_model_parallel_size * self.expert_model_parallel_size()

        @property
        def data_parallel_size(self):
            """
            :meta private:
            """
            return mpu.get_data_parallel_world_size()

        @property
        def data_parallel_rank(self):
            """
            :meta private:
            """
            return mpu.get_data_parallel_rank()

        def pipeline_parallel_rank(self):
            """
            :meta private:
            """
            return mpu.get_pipeline_model_parallel_rank()

        def tensor_parallel_rank(self):
            """
            :meta private:
            """
            return mpu.get_tensor_model_parallel_rank()

        def tensor_and_expert_parallel_group(self):
            """
            :meta private:
            """
            return mpu.get_tensor_and_expert_parallel_group()

        def expert_parallel_rank(self):
            """
            :meta private:
            """
            if hasattr(mpu, "get_expert_model_parallel_rank"):
                return mpu.get_expert_model_parallel_rank()
            return 0

        def num_layers(self):
            """
            :meta private:
            """
            return self.megatron_args.num_layers

        def megatron_model(self):
            if isinstance(self.model, list):
                assert len(self.model) == 1
                model = self.model[0]
            else:
                model = self.model
            return model

        def build_pipeline_layer_name_mapping(self, num_target_pipe_stage, target_pipe_rank, tgt_layer_offset, requires_grad=True):
            # pylint: disable=unused-argument
            """
            remap pipeline layer_name. For each pipeline stage, the layer number starts with 0.
            Args:
                num_target_pipe_stage: number of pipeline stage in target model
                target_pipe_rank: target model pipeline rank
                tgt_layer_offset: target model pipeline stage layer offset
                requires_grad: (deprecated) unused

            :meta private:
            """
            src_layer_offset = self.get_pipeline_stage_layer_offset()
            model = self.megatron_model()
            is_tgt_last_stage = target_pipe_rank == num_target_pipe_stage - 1 and target_pipe_rank != 0
            name_mapping = {}
            for src_name, _ in model.named_parameters():
                if src_name.endswith("word_embeddings.weight") \
                        and "language_model" not in src_name \
                        and hasattr(unwrap_model(model), "language_model"):
                    # See comment in MegatronModule.initialize_word_embeddings()
                    if not is_tgt_last_stage:
                        tgt_name = src_name.replace("word_embeddings.weight", "language_model.embedding.word_embeddings.weight")
                    else:
                        tgt_name = src_name
                else:
                    # Translate destination layer number (0-N for each partition)
                    # to source layer number (single-model layer number)
                    # e.g. for src model with 8 layers, src_num_stage=4, dst_num_stage=2
                    # for src_model, stage offsets are [0, 2, 4, 6]. for dst model, stage offsets are [0, 4]
                    # then the start layer_num of src->dst is as follows:
                    # stage0 0->0 stage1 0->(2-0) stage2 0->(4-4) stage3 0->(6-4)
                    start_layer_num = src_layer_offset - tgt_layer_offset
                    _update_layer_num = functools.partial(update_layer_num, start_layer_num)
                    tgt_name = re.sub(layer_re, _update_layer_num, src_name)
                name_mapping[tgt_name] = src_name

            for src_name, _ in model.named_buffers():
                if 'local_tokens_per' in src_name:
                    continue
                start_layer_num = src_layer_offset - tgt_layer_offset
                _update_layer_num = functools.partial(update_layer_num, start_layer_num)
                tgt_name = re.sub(layer_re, _update_layer_num, src_name)
                name_mapping[tgt_name] = src_name

            return name_mapping

        def get_local_param_ranks(self):
            """
            :meta private:
            """
            if self.expert_model_parallel_size() == 1:
                data_parallel_global_ranks = list(mpu._DATA_PARALLEL_GLOBAL_RANKS)
                return data_parallel_global_ranks, mpu.get_data_parallel_rank()
            else:
                # Get data parallel modulo expert parallel ranks
                # NOTE: for compatability, use `get_expert_data_parallel_group` instead of
                # `get_data_modulo_expert_parallel_group` if possible
                if hasattr(mpu, 'get_expert_data_parallel_group'):
                    data_modulo_expert_parallel_group = mpu.get_expert_data_parallel_group()
                    data_modulo_expert_parallel_ranks = dist.get_process_group_ranks(data_modulo_expert_parallel_group)
                    this_rank = mpu.get_expert_data_parallel_rank()
                else:
                    data_modulo_expert_parallel_group = mpu.get_data_modulo_expert_parallel_group()
                    data_modulo_expert_parallel_ranks = dist.get_process_group_ranks(data_modulo_expert_parallel_group)
                    this_rank = mpu.get_data_modulo_expert_parallel_rank()
                return data_modulo_expert_parallel_ranks, this_rank

        def save_checkpoint(self, iteration):
            """
            save checkpoint at `iteration`
            :param iteration: save iteration

            :meta private:
            """
            save_checkpoint_and_time(
                iteration,
                self.model,
                self.optimizer,
                self.opt_param_scheduler,
                0,
                None
            )


        def offload_optimizer_states(self):
            """
            offload optimizer states
            """
            if self.module_args.free_gpu_memory.offload_optimizer_states:
                self._memory_manager.offload_optimizer_states()

        def onload_optimizer_states(self):
            """
            onload optimizer states
            """
            if self.module_args.free_gpu_memory.offload_optimizer_states:
                self._memory_manager.onload_optimizer_states()

        def offload_main_weights(self):
            """
            offload main weights
            """
            if self.module_args.free_gpu_memory.offload_weights:
                self._memory_manager.offload_main_weights()

        def onload_main_weights(self):
            """
            onload main weights
            """
            if self.module_args.free_gpu_memory.offload_weights:
                self._memory_manager.onload_main_weights()

        def offload_weights(self):
            """
            offload weights
            """
            if self.module_args.free_gpu_memory.offload_weights:
                self._memory_manager.offload_weights()

        def onload_weights(self):
            """
            onload weights
            """
            if self.module_args.free_gpu_memory.offload_weights:
                self._memory_manager.onload_weights()

        def free_grad_buffers(self):
            """
            free grad buffers and related tensors
            """
            if self.module_args.free_gpu_memory.free_grad_buffers:
                self._memory_manager.free_grad_buffers()

        def build_grad_buffers(self):
            """
            build grad buffers and related tensors
            """
            if self.module_args.free_gpu_memory.free_grad_buffers:
                self._memory_manager.build_grad_buffers()

        def get_pipeline_stage_layer_num(self):
            assert self.stage2layer_num is not None
            if self.stage2layer_num[self.pipeline_parallel_rank()] is not None:
                return self.stage2layer_num[self.pipeline_parallel_rank()]
            layer_re = re.compile(r'layers\.([0-9]+)')
            layer_set = set()
            for name in self.named_parameters:
                layer_num = re.findall(layer_re, name)
                if layer_num:
                    layer_set.add(layer_num[0])
            stage_layer_num = len(layer_set)
            return stage_layer_num

        def get_pipeline_stage_layer_offset(self):
            assert self.stage2offset is not None and \
                self.stage2offset[self.pipeline_parallel_rank()] is not None
            return self.stage2offset[self.pipeline_parallel_rank()]

        @torch.no_grad()
        def collect_sparse_params(self):
            from megatron.core.parallel_state import ( # pylint: disable=import-outside-toplevel
                get_expert_model_parallel_group,
                get_expert_model_parallel_world_size
            )
            to_be_merged = []
            for name, params_to_sync in self.named_parameters.items():
                if 'mlp.experts.linear_fc1' in name or 'mlp.experts.linear_fc2' in name:
                    to_be_merged.append([name, params_to_sync])
            to_be_merged = sorted(to_be_merged, key=lambda x: x[0])
            self._sparse_params = {}
            for name, params_to_sync in to_be_merged:
                w, h = params_to_sync.shape
                out_tensor = torch.empty(
                    [get_expert_model_parallel_world_size(), w, h],
                    dtype=params_to_sync.dtype,
                    device=params_to_sync.device
                )
                dist.all_gather_into_tensor(out_tensor, params_to_sync, group=get_expert_model_parallel_group())
                self._sparse_params[name] = out_tensor

else:
    class MegatronModule(TorchModule):
        """Module Placeholder for Megatron Backend"""
        def __init__(self, *args, **kwargs):
            # pylint: disable=super-init-not-called
            raise SystemError("Cannot import megatron backend, please check your environment variable.")
