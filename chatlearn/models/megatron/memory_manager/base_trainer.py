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
"""Base class and creator function for trainer memory managers."""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer
from megatron.core.optimizer import (
    DistributedOptimizer,
    Float16OptimizerWithFloat16Params,
)

from chatlearn.utils.flat_tensors import BucketizedFlatTensors
from chatlearn.utils.logger import log_rank_0


class BaseTrainerMemoryManager(ABC):
    """
    Base class for Megatron trainer memory managers, which provides common routines for all versions, such as
    optimizer states offloading, and main weights offloading.
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
        self._model = model
        self._optimizer = optimizer
        self._accumulate_allreduce_grads_in_fp32 = accumulate_allreduce_grads_in_fp32
        self._params_dtype = params_dtype
        self._use_distributed_optimizer = use_distributed_optimizer
        self._bucket_size_mb = bucket_size_mb

        assert isinstance(
            model, (DistributedDataParallel,)
        ), f'Only support model type DistributedDataParallel, current type is {str(type(model))}.'
        sub_optimizers = [optimizer]
        from megatron.core.optimizer.optimizer import ChainedOptimizer # pylint: disable=import-outside-toplevel
        assert isinstance(
            optimizer, (MixedPrecisionOptimizer, ChainedOptimizer)
        ), f'Only support optimizer type MixedPrecisionOptimizer, ChainedOptimizer and its subclasses, current type is {str(type(optimizer))}.'
        if isinstance(optimizer, ChainedOptimizer):
            sub_optimizers = optimizer.chained_optimizers

        for optim in sub_optimizers:
            # sanity check
            if self._use_distributed_optimizer:
                assert isinstance(optim, DistributedOptimizer)
            else:
                log_rank_0('Current optimizer is Float16OptimizerWithFloat16Params')
                assert isinstance(optim, Float16OptimizerWithFloat16Params)

        self._main_weights_offloaded = False
        self._group_flat_main_weights: Optional[List[BucketizedFlatTensors]] = None

    def _optimizer_load_state_bucket_into_device(self, device):
        """put the state bucket onto a device"""
        state_dict = self._optimizer.optimizer.state_dict()
        for tensors in state_dict['state'].values():
            keys = list(tensors.keys())
            for key in keys:
                # compatible with transformer_engine v1.10, state['master_param']=None
                if tensors[key] is not None:
                    tensors[key] = tensors[key].to(device=device, non_blocking=True)
        # make sure the loading is finished before returning
        torch.cuda.synchronize()

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

    def _flat_param_groups(self, multi_groups: List[List[List[torch.Tensor]]]):
        """
        Flatten parameters in param groups.
        """
        return [
            BucketizedFlatTensors(group, primary_store_device='cpu', bucket_size_mb=self._bucket_size_mb)
            for groups in multi_groups
            for group in groups
        ]

    def offload_main_weights(self):
        """
        offload main weights
        """
        if self._main_weights_offloaded:
            log_rank_0('Call offload_main_weights when already offloaded. Ignore it.')
            return

        if self._group_flat_main_weights is None:
            if self._use_distributed_optimizer:
                self._group_flat_main_weights = self._flat_param_groups(
                    [self._optimizer.shard_fp32_from_float16_groups]
                )
            else:
                self._group_flat_main_weights = self._flat_param_groups([self._optimizer.fp32_from_float16_groups])

        for flat_main_weights in self._group_flat_main_weights:
            flat_main_weights.copy_to_primary_store()

        self._main_weights_offloaded = True

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

    @abstractmethod
    def offload_weights(self):
        """
        offload weights
        """

    @abstractmethod
    def onload_weights(self):
        """
        onload weights
        """

    @abstractmethod
    def free_grad_buffers(self):
        """
        free grad buffers and related tensors
        """

    @abstractmethod
    def build_grad_buffers(self):
        """
        build grad buffers and related tensors
        """
