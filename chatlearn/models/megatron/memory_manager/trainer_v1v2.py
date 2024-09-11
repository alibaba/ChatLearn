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
"""Trainer Memery manager for Megatron V1 and V2"""
from chatlearn.utils.megatron_import_memory_helper import MegatronVersion, check_megatron_versions

check_megatron_versions([MegatronVersion.V1, MegatronVersion.V2])

# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
from typing import List, Optional

import torch

from chatlearn.models.megatron.memory_manager.base_trainer import BaseTrainerMemoryManager
from chatlearn.utils.flat_tensors import BucketizedFlatTensors
from chatlearn.utils.logger import log_rank_0
from chatlearn.utils.megatron_import_helper import tensor_parallel

# pylint: enable=wrong-import-position,wrong-import-order,ungrouped-imports

__all__ = ['TrainerMemoryManagerV1V2']


class TrainerMemoryManagerV1V2(BaseTrainerMemoryManager):
    """
    Memory manager for Megatron V1 and V2 trainer modules.
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

        self._grad_dtype_to_params = self._get_grad_dtype_to_params(model, accumulate_allreduce_grads_in_fp32)

        self._group_flat_weights: Optional[List[BucketizedFlatTensors]] = None
        self._grad_buffers_numels = None
        self._grad_buffers_bucket_sizes = None

    def get_grad_buffers(self):
        if self._megatron_version == MegatronVersion.V2:
            return self._model.grad_buffers
        elif self._megatron_version == MegatronVersion.V1:
            return self._model._grad_buffers

    @staticmethod
    def _get_grad_dtype_to_params(model, accumulate_allreduce_grads_in_fp32):
        # Group parameters by their gradient type.
        grad_dtype_to_params = {}
        for _, param in model.module.named_parameters():
            if param.requires_grad and getattr(param, 'allreduce', True):
                param.grad_added_to_main_grad = False
                dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype
                params = grad_dtype_to_params.get(dtype, [])
                params.append(param)
                grad_dtype_to_params[dtype] = params
        return grad_dtype_to_params

    def offload_weights(self):
        """
        offload weights
        """
        if self._weights_offloaded:
            log_rank_0('Call offload_weights when already offloaded. Ignore it.')
            return

        optimizer = self._optimizer

        if self._use_distributed_optimizer:
            optimizer.shard_float16_groups.clear()
            optimizer.shard_fp32_groups.clear()

        if self._group_flat_weights is None:
            if self._use_distributed_optimizer:
                self._group_flat_weights = self._flat_param_groups(
                    [
                        optimizer.model_float16_groups,
                        optimizer.model_fp32_groups,
                    ],
                )
            else:
                self._group_flat_weights = self._flat_param_groups(
                    [
                        optimizer.float16_groups,
                        optimizer.fp32_from_fp32_groups,
                    ],
                )

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

        for flat_weights in self._group_flat_weights:
            flat_weights.copy_to_gpu_buffer()

        model = self._model
        # Re-register grad acc hooks, see Megatron DistributedDataParallel#__init__.
        model.grad_accs = []
        for param in model.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                if self._megatron_version == MegatronVersion.V2:
                    grad_acc.register_hook(model._make_param_hook(param, model.param_to_grad_buffer))
                elif self._megatron_version == MegatronVersion.V1:
                    grad_acc.register_hook(model._make_param_hook(param))
                model.grad_accs.append(grad_acc)

        if not self._use_distributed_optimizer:
            self._weights_offloaded = False
            return

        shard_float16_groups = optimizer.shard_float16_groups
        shard_fp32_groups = optimizer.shard_fp32_groups
        param_gbuf_map = optimizer.model_param_gbuf_map
        opt_group_ranges = optimizer.opt_group_ranges
        model_gbuf_ranges = optimizer.model_gbuf_ranges

        # Rebuild shard_float16_groups and shard_fp32_groups,
        # see Megatron DistributedOptimizer#build_model_and_main_param_groups.
        for _, group_range in enumerate(opt_group_ranges):
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)

            for model_param in group_range["params"]:
                assert model_param.requires_grad
                if self._megatron_version == MegatronVersion.V2:
                    model_index, dtype, bucket_index = param_gbuf_map[model_param]
                    gbuf_range = model_gbuf_ranges[model_index][dtype][bucket_index]
                    param_range = gbuf_range["param_map"][model_param]["param"]
                elif self._megatron_version == MegatronVersion.V1:
                    model_index, dtype = param_gbuf_map[model_param]
                    gbuf_range = model_gbuf_ranges[model_index][dtype]
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
        grad_dtype_to_params = self._grad_dtype_to_params

        # This is necessary, but don't know why.
        optimizer.zero_grad(True)

        if self._use_distributed_optimizer:
            # Release param_buffers because they share storage with grad_buffers.
            # Note: param_buffers are only available in DistributedOptimizer.
            optimizer.param_buffers.clear()

        # Release grad_buffers, including buckets in GradBuffer for newer Megatron version.
        # Release `main_grad` of parameters.
        self._grad_buffers_numels = {}
        self._grad_buffers_bucket_sizes = {}

        for dtype, buffer in self.get_grad_buffers().items():
            for p in grad_dtype_to_params[dtype]:
                del p.main_grad

            self._grad_buffers_numels[dtype] = buffer.numel_padded

            if self._megatron_version == MegatronVersion.V2:
                bucket_sizes = []
                for bucket in buffer.buckets:
                    bucket_sizes.append(bucket.data.numel())
                    bucket.data = None
                self._grad_buffers_bucket_sizes[dtype] = bucket_sizes

            buffer.data = None

        self._grad_buffers_freed = True

    def build_grad_buffers(self):
        """
        build grad buffers and related tensors
        """
        if not self._grad_buffers_freed:
            log_rank_0('Call build_grad_buffers when already built. Ignore it.')
            return

        optimizer = self._optimizer
        params_dtype = self._params_dtype
        grad_dtype_to_params = self._grad_dtype_to_params

        # Re-allocate data of grad_buffers, including data of buckets, see Megatron DistributedDataParallel#__init__.
        # Also set `main_grad` for parameters.
        for dtype, buffer in self.get_grad_buffers().items():
            numel_padded = self._grad_buffers_numels[dtype]
            buffer.data = torch.zeros(
                numel_padded,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

            if self._megatron_version == MegatronVersion.V2:
                for bucket, numel in zip(buffer.buckets, self._grad_buffers_bucket_sizes[dtype]):
                    bucket.data = buffer.get(torch.Size([numel]), bucket.offset)

            params = grad_dtype_to_params[dtype]
            data_start_index = 0
            for param in params[::-1]:
                if not param.requires_grad:
                    continue
                this_numel = param.data.nelement()
                data_end_index = data_start_index + this_numel
                param.main_grad = buffer.get(param.data.shape, data_start_index)
                data_start_index = data_end_index

        if not self._use_distributed_optimizer:
            self._grad_buffers_freed = False
            return

        # Re-allocate param_buffers, see Megatron DistributedOptimizer#__init__.
        optimizer.param_buffers = []
        for _, _ in enumerate(optimizer.models):
            current_param_buffers = {}
            for dtype, grad_buffer in self.get_grad_buffers().items():
                current_param_buffers[dtype] = []
                if self._megatron_version == MegatronVersion.V2:
                    for bucket in grad_buffer.buckets:
                        try:
                            storage = bucket.data.storage()._untyped()
                        # pylint: disable-next=bare-except
                        except:
                            storage = bucket.data.storage().untyped()

                        param_buffer = torch.tensor([], dtype=params_dtype, device=bucket.data.device).set_(storage)
                        param_buffer = param_buffer[bucket.offset : bucket.offset + bucket.data.numel()]
                        current_param_buffers[dtype].append(param_buffer)
                elif self._megatron_version == MegatronVersion.V1:
                    try:
                        storage = grad_buffer.data.storage()._untyped()
                    # pylint: disable-next=bare-except
                    except:
                        storage = grad_buffer.data.storage().untyped()
                    param_buffer = torch.tensor([], dtype=params_dtype, device=grad_buffer.data.device).set_(storage)
                    param_buffer = param_buffer[: grad_buffer.numel_padded]
                    current_param_buffers[dtype] = param_buffer
            optimizer.param_buffers.append(current_param_buffers)

        self._grad_buffers_freed = False
