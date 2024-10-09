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
"""Trainer Memery manager for Megatron V3"""
from typing import List, Optional

import torch

from chatlearn.models.megatron.memory_manager.base_trainer import BaseTrainerMemoryManager
from chatlearn.utils.flat_tensors import BucketizedFlatTensors, FlatTensors
from chatlearn.utils.logger import log_rank_0
from chatlearn.utils.megatron_import_helper import tensor_parallel
from chatlearn.utils.megatron_import_memory_helper import BufferType
from chatlearn.utils.megatron_import_memory_helper import MegatronVersion, check_megatron_versions

check_megatron_versions([MegatronVersion.V3])


__all__ = ['TrainerMemoryManagerV3']


class TrainerMemoryManagerV3(BaseTrainerMemoryManager):
    """
    Memory manager for Megatron V3 trainer modules.
    """

    def __init__(
        self,
        model,
        optimizer,
        use_distributed_optimizer,
        accumulate_allreduce_grads_in_fp32,
        params_dtype,
        bucket_size_mb=0,
    ):
        super().__init__(
            model,
            optimizer,
            use_distributed_optimizer,
            accumulate_allreduce_grads_in_fp32,
            params_dtype,
            bucket_size_mb,
        )
        self._weights_offloaded = False
        self._grad_buffers_freed = False

        self._buffers = self._get_buffers(model)

        self._group_flat_weights: Optional[List[BucketizedFlatTensors]] = None

    @staticmethod
    def _get_buffers(model):
        processed_buffers = set()
        buffers = []
        for _, buffer in model.param_to_buffer.items():
            if buffer not in processed_buffers:
                processed_buffers = set()
                processed_buffers.add(buffer)
                buffers.append(buffer)
        return buffers

    def offload_weights(self):
        """
        offload weights
        """
        if self._weights_offloaded:
            log_rank_0('Call offload_weights when already offloaded. Ignore it.')
            return

        optimizer = self._optimizer

        # TODO(jiqi): support expert parallel params

        # In the V3 version, when distributed optimizer is used, parameter data are managed together with
        # gradients in buffers.
        if self._use_distributed_optimizer:
            optimizer.shard_float16_groups.clear()
            optimizer.shard_fp32_groups.clear()
            optimizer.pbuf_view_items.clear()

            if self._group_flat_weights is None:
                self._group_flat_weights = []
                for buffer in self._buffers:
                    assert buffer.param_data is not None
                    self._group_flat_weights.append(
                        BucketizedFlatTensors([buffer.param_data], self._bucket_size_mb, 'cpu')
                    )

            # Remove references from params
            for p, _ in self._model.param_to_buffer.items():
                # save the shape for reconstruction
                p._saved_shape = p.shape
                p.data = FlatTensors._EMPTY_TENSOR

            # Remove references from buckets
            for buffer in self._buffers:
                for bucket in buffer.buckets:
                    bucket.param_data = None
        else:
            if self._group_flat_weights is None:
                self._group_flat_weights = self._flat_param_groups(
                    [
                        optimizer.float16_groups,
                        optimizer.fp32_from_fp32_groups,
                    ],
                )

        # Offload param_data of buffers
        for flat_weights in self._group_flat_weights:
            flat_weights.copy_to_primary_store()

        self._model.grad_accs.clear()

        self._weights_offloaded = True

    def onload_weights(self):
        """
        onload weights
        """
        if not self._weights_offloaded:
            log_rank_0('Call onload_weights when already onloaded. Ignore it.')
            return

        optimizer = self._optimizer

        # Onload param_data of buffers
        for flat_weights in self._group_flat_weights:
            flat_weights.copy_to_gpu_buffer()

        if self._use_distributed_optimizer:
            # Reconstruct references from buckets
            for buffer in self._buffers:
                assert buffer.param_data is not None
                for bucket_id, bucket in enumerate(buffer.buckets):
                    (start_index, end_index) = buffer.bucket_indices[bucket_id]
                    bucket.param_data = None
                    if buffer.param_data is not None:
                        bucket.param_data = buffer._get(
                            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
                        )

            # Reconstruct references from params
            for param, buffer in self._model.param_to_buffer.items():
                data_start_index, _, bucket_id = buffer.param_index_map[param]
                if buffer.param_data is not None:
                    param.data = buffer._get(param._saved_shape, data_start_index, buffer_type=BufferType.PARAM)

        model = self._model
        # Re-register grad acc hooks, see Megatron DistributedDataParallel#__init__.
        model.grad_accs = []
        for param in model.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(model._make_param_hook(param, model.param_to_buffer))
                model.grad_accs.append(grad_acc)

        if not self._use_distributed_optimizer:
            self._weights_offloaded = False
            return

        optimizer.pbuf_view_items = optimizer._get_model_param_buffer_dp_views()

        shard_float16_groups = optimizer.shard_float16_groups
        shard_fp32_groups = optimizer.shard_fp32_groups
        param_gbuf_map = optimizer.model_param_gbuf_map
        opt_group_ranges = optimizer.opt_group_ranges
        model_gbuf_ranges = optimizer.gbuf_ranges

        # Rebuild shard_float16_groups and shard_fp32_groups,
        # see Megatron DistributedOptimizer#build_model_and_main_param_groups.
        for _, group_range in enumerate(opt_group_ranges):
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)

            for model_param in group_range["params"]:
                assert model_param.requires_grad
                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = model_gbuf_ranges[gbuf_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]

                # fp16, bf16 params.
                if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                    shard_model_param = model_param.detach().view(-1)[param_range.start : param_range.end]
                    tensor_parallel.copy_tensor_model_parallel_attributes(shard_model_param, model_param)
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared

                    shard_float16_params_this_group.append(shard_model_param)

                # fp32 params.
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
                    shard_fp32_params_this_group.append(shard_model_param)
                    tensor_parallel.copy_tensor_model_parallel_attributes(shard_model_param, model_param)
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared
                else:
                    raise TypeError(
                        'Wrapped parameters must be one of '
                        'torch.cuda.FloatTensor,  '
                        'torch.cuda.HalfTensor, or '
                        'torch.cuda.BFloat16Tensor. '
                        'Received {}'.format(model_param.type())
                    )

        self._weights_offloaded = False

    def free_grad_buffers(self):
        """
        free grad buffers and related tensors
        """
        if self._grad_buffers_freed:
            log_rank_0('Call free_grad_buffers when already freed. Ignore it.')
            return

        optimizer = self._optimizer

        # This is necessary, but don't know why.
        optimizer.zero_grad(True)

        # Remove references from params
        for p, buffer in self._model.param_to_buffer.items():
            del p.main_grad

        # Remove references from buckets and free grad_data of buffer
        for buffer in self._buffers:
            for bucket in buffer.buckets:
                del bucket.grad_data
            del buffer.grad_data

        self._grad_buffers_freed = True

    def build_grad_buffers(self):
        """
        build grad buffers and related tensors
        """
        if not self._grad_buffers_freed:
            log_rank_0('Call build_grad_buffers when already built. Ignore it.')
            return

        # Build buffers and reconstruct references from buckets
        for buffer in self._buffers:
            buffer.grad_data = torch.zeros(
                buffer.numel,
                dtype=buffer.grad_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            for bucket_id, bucket in enumerate(buffer.buckets):
                (start_index, end_index) = buffer.bucket_indices[bucket_id]
                bucket.grad_data = buffer._get(
                    torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
                )

        # Reconstruct references from params
        for param, buffer in self._model.param_to_buffer.items():
            data_start_index, _, bucket_id = buffer.param_index_map[param]
            param.main_grad = buffer._get(param.data.shape, data_start_index, buffer_type=BufferType.GRAD)

        self._grad_buffers_freed = False
