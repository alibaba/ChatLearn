# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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
"""Trainer Memery manager for Megatron-Core"""
from typing import List, Optional, Union

import torch

from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core import tensor_parallel
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from megatron.core.optimizer.optimizer import (
    MixedPrecisionOptimizer,
    ChainedOptimizer
)
from megatron.core.distributed.param_and_grad_buffer import BufferType
from megatron.core.optimizer import (
    DistributedOptimizer,
    Float16OptimizerWithFloat16Params,
)

from chatlearn.utils.flat_tensors import BucketizedFlatTensors, FlatTensors
from chatlearn.utils.logger import log_rank_0


__all__ = ['TrainerMemoryManager']


class TrainerMemoryManager:
    """
    Memory manager for Megatron training modules which provides utilities to free memory when unused.
    """
    def __init__(
        self,
        model: List[DistributedDataParallel],
        optimizer: Union[MixedPrecisionOptimizer, ChainedOptimizer],
        bucket_size_mb: int=0,
    ):
        """Manage memory of training models with one or more chunks.

        Args:
            model (List[nn.Module]): The Megatron model chunks to be managed. 
            optimizer (Union[MixedPrecisionOptimizer, ChainedOptimizer]): The optimizer
            to be managed. Should be (chained or individual) FP16Optimizer or 
            DistributedOptimizer.
            bucket_size_mb (int): The bucket size when offloading weights.
        """
        self._model = model
        self._optimizer = optimizer
        self._use_distributed_optimizer = self._check_sanity()
        self._bucket_size_mb = bucket_size_mb

        self._main_weights_offloaded = False
        self._group_flat_main_weights: Optional[List[BucketizedFlatTensors]] = None
        self._weights_offloaded = False
        self._grad_buffers_freed = False

        # NOTE: Though ParamAndGradBuffer in Megatron-Core is refactorized into an
        # internal class. In MemoryManager, we have to do some modification on their
        # data.
        self._buffers = self._get_buffers(model)
        self._group_flat_weights: Optional[List[BucketizedFlatTensors]] = None

    def _flat_param_groups(self, multi_groups: List[List[List[torch.Tensor]]]):
        """
        Flatten parameters in param groups.
        """
        return [
            BucketizedFlatTensors(group, primary_store_device='cpu', bucket_size_mb=self._bucket_size_mb)
            for groups in multi_groups
            for group in groups
        ]

    def _check_sanity(self):
        """Check whether the model and optimizer are supported in this memory
        manager.
        """
        if any(not isinstance(model_chunk, (DistributedDataParallel,)) for model_chunk in self._model):
            all_types = ','.join([str(type(model_chunk)) for model_chunk in self._model] )
            raise NotImplementedError(f'Only support model type DistributedDataParallel, current type is {all_types}.')

        assert isinstance(
            self._optimizer, (MixedPrecisionOptimizer, ChainedOptimizer)
        ), f'Only support optimizer type MixedPrecisionOptimizer, ChainedOptimizer and its subclasses, current type is {str(type(optimizer))}.'

        sub_optimizers = self._get_sub_optimizers()
        if len(sub_optimizers) == 0:
            return True # should not return here (though supported)

        assert all(isinstance(optim, type(sub_optimizers[0])) for optim in sub_optimizers), \
            "All sub optimizers should be the same type"
        use_distributed_optimizer = isinstance(sub_optimizers[0], DistributedOptimizer)
        if not use_distributed_optimizer:
            assert isinstance(sub_optimizers[0], Float16OptimizerWithFloat16Params), \
                "Only Float16OptimizerWithFloat16Params is supported when distributed optimizer is not used"
        return use_distributed_optimizer

    @staticmethod
    def _get_buffers(model):
        """
        Get the unique _ParamAndGradBuffer from DDP model.

        """
        # DDP: buffers & expert_parallel_buffers
        processed_buffers = set()
        buffers = []
        for model_chunk in model:
            for buffer in model_chunk.buffers:
                if buffer not in processed_buffers:
                    processed_buffers.add(buffer)
                    buffers.append(buffer)
            for buffer in model_chunk.expert_parallel_buffers:
                if buffer not in processed_buffers:
                    processed_buffers.add(buffer)
                    buffers.append(buffer)
        return buffers

    def param_to_buffer(self):
        param_to_buffer = {}
        for buffer in self._buffers:
            for param in buffer.params:
                param_to_buffer[param] = buffer
        return param_to_buffer

    def _get_sub_optimizers(self):
        if isinstance(self._optimizer, ChainedOptimizer):
            return self._optimizer.chained_optimizers
        else:
            return [self._optimizer]

    def _optimizer_load_state_bucket_into_device(self, device):
        """put the state bucket onto a device"""
        for sub_optimizer in self._get_sub_optimizers():
            # NOTE: compatible with transformer_engine v1.13, in-place offload the origin state dict
            # If we use sub_optimizer.optimizer.state_dict(), we'll get a deepcopy of state dict instead
            state_dict = torch.optim.Optimizer.state_dict(sub_optimizer.optimizer)
            for tensors in state_dict['state'].values():
                keys = list(tensors.keys())
                for key in keys:
                    # compatible with transformer_engine v1.10, state['master_param']=None
                    if tensors[key] is not None:
                        tensors[key] = tensors[key].to(device=device, non_blocking=True)
        # make sure the loading is finished before returning
        torch.cuda.synchronize()

    def offload_weights(self):
        """
        offload weights
        """
        if self._weights_offloaded:
            log_rank_0('Call offload_weights when already offloaded. Ignore it.')
            return

        if self._use_distributed_optimizer:
            for optimizer in self._get_sub_optimizers():
                optimizer.shard_float16_groups.clear()
                optimizer.shard_fp32_groups.clear()
                if hasattr(optimizer, 'pbuf_view_items'):
                    optimizer.pbuf_view_items.clear()

                if self._group_flat_weights is None:
                    self._group_flat_weights = []
                    for buffer in self._buffers:
                        assert buffer.param_data is not None
                        self._group_flat_weights.append(
                            BucketizedFlatTensors([buffer.param_data], self._bucket_size_mb, 'cpu')
                        )

            # Remove references from params
            for model_chunk in self._model:
                for p in model_chunk.parameters():
                    # save the shape for reconstruction
                    p._saved_shape = p.shape
                    p.data = FlatTensors._EMPTY_TENSOR

            # Remove references from buckets
            for buffer in self._buffers:
                for bucket in buffer.buckets:
                    bucket.param_data = None

        elif self._group_flat_weights is None:
            optimizer_groups = []
            for optimizer in self._get_sub_optimizers():
                optimizer_groups.extend([
                    optimizer.float16_groups,
                    optimizer.fp32_from_fp32_groups,
                ])
            self._group_flat_weights = self._flat_param_groups(optimizer_groups)

        # Offload param_data of buffers
        for flat_weights in self._group_flat_weights:
            flat_weights.copy_to_primary_store()

        for model_chunk in self._model:
            model_chunk.grad_accs.clear()

        self._weights_offloaded = True

    def onload_weights(self):
        """
        onload weights
        """
        if not self._weights_offloaded:
            log_rank_0('Call onload_weights when already onloaded. Ignore it.')
            return

        # Onload param_data of buffers
        for flat_weights in self._group_flat_weights:
            flat_weights.copy_to_gpu_buffer()

        for optimizer in self._get_sub_optimizers():
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
                for param, buffer in self.param_to_buffer().items():
                    data_start_index, _, bucket_id = buffer.param_index_map[param]
                    if buffer.param_data is not None:
                        param.data = buffer._get(param._saved_shape, data_start_index, buffer_type=BufferType.PARAM)

        for model_chunk in self._model:
            # Re-register grad acc hooks, see Megatron DistributedDataParallel#__init__.
            model_chunk.grad_accs = []
            for param in model_chunk.module.parameters():
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator function.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    # NOTE: Since Megatron-LM COMMIT 655a663, _make_param_hook is renamed to _make_backward_post_hook
                    try:
                        grad_acc.register_hook(model_chunk._make_param_hook(param, model_chunk.param_to_buffer))
                    except AttributeError:
                        grad_acc.register_hook(model_chunk._make_backward_post_hook(param))
                    model_chunk.grad_accs.append(grad_acc)

        if not self._use_distributed_optimizer:
            self._weights_offloaded = False
            return

        for optimizer in self._get_sub_optimizers():
            # NOTE: Since Megatron-LM COMMIT 655a663, pbuf_view_items is dropped
            if hasattr(optimizer, 'pbuf_view_items'):
                assert hasattr(optimizer, '_get_model_param_buffer_dp_views')
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

        # NOTE: detach grad in params of ChainedOptimizer / Float16Optimizer
        self._optimizer.zero_grad(True)

        # NOTE: for MoE model, detach probs in the token dispatcher to avoid memory leak
        # TODO: MCore will provide a fix for this issue in the future, remove this when fixed
        for model_chunk in self._model:
            for _, m in model_chunk.named_modules():
                if isinstance(m, MoELayer):
                    m.token_dispatcher.probs, m.token_dispatcher.routing_map = None, None

        # NOTE: delete main_grad in params of ChainedOptimizer / Float16Optimizer
        for p, buffer in self.param_to_buffer().items():
            del p.main_grad

        # NOTE: Remove references from buckets and free grad_data of buffer
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
        for param, buffer in self.param_to_buffer().items():
            data_start_index, _, bucket_id = buffer.param_index_map[param]
            param.main_grad = buffer._get(param.data.shape, data_start_index, buffer_type=BufferType.GRAD)

        self._grad_buffers_freed = False

    def offload_main_weights(self):
        """
        offload main weights
        """
        if self._main_weights_offloaded:
            log_rank_0('Call offload_main_weights when already offloaded. Ignore it.')
            return

        if self._group_flat_main_weights is None:
            optimizer_groups = []
            for optimizer in self._get_sub_optimizers():
                if self._use_distributed_optimizer:
                    optimizer_groups.extend([optimizer.shard_fp32_from_float16_groups])
                else:
                    optimizer_groups.extend([optimizer.fp32_from_float16_groups])
            self._group_flat_main_weights = self._flat_param_groups(optimizer_groups)

        for flat_main_weights in self._group_flat_main_weights:
            flat_main_weights.copy_to_primary_store()

        self._main_weights_offloaded = True

    def offload_optimizer_states(self):
        """
        offload optimizer states
        """
        self._optimizer_load_state_bucket_into_device(device='cpu')

    def onload_optimizer_states(self):
        """
        onload optimizer states
        """
        self._optimizer_load_state_bucket_into_device(device=torch.cuda.current_device())

    def onload_main_weights(self):
        """
        onload weights and allocate grads
        """
        if not self._main_weights_offloaded:
            log_rank_0('Call onload_main_weights when already onloaded. Ignore it.')
            return

        for flat_main_weights in self._group_flat_main_weights:
            flat_main_weights.copy_to_gpu_buffer()
        self._main_weights_offloaded = False
