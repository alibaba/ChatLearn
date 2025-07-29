# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
import math
from dataclasses import fields
import torch

try:
    from megatron.training import get_args
    from megatron.training.arguments import parse_args
    from megatron.training.utils import unwrap_model
    from megatron.core import parallel_state as mpu
    from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
    from megatron.training.training import save_checkpoint_and_time
    IS_MEGATRON_SUPPORTED = True
except ImportError:
    IS_MEGATRON_SUPPORTED = False

from chatlearn.configs.common import BaseConfig
from chatlearn.utils.mappings import build_sharded_info_for_mcore_model
from .torch_module import TorchModule


if IS_MEGATRON_SUPPORTED:
    try:
        # pylint: disable-next=import-outside-toplevel, unused-import, ungrouped-imports
        from megatron.core.distributed.distributed_data_parallel import _ParamAndGradBuffer
    except ImportError as exc:
        raise ValueError(
            'Old or customed version of Megatron is no longer supported. Please checkout to 0f4e0e1872b62a96d0465de477f26ae81a2e33d7'
        ) from exc
    # pylint: disable-next=ungrouped-imports
    from chatlearn.models.megatron.memory_manager import InferenceMemoryManager, TrainerMemoryManager
    # pylint: disable-next=ungrouped-imports
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    class MegatronModule(TorchModule):
        """MegatronModule is the class for Alignment Megatron models."""

        def __init__(self, name: str, args=None, replica_id: int=0):
            """The chatlearn wrapper for a Megatron model.

            Args:
                name (str): The name of this module
                args (Any, optional): The arguments. Defaults to None.
                replica_id (int, optional): The replica id of this module. Defaults to 0.
            """
            super().__init__(name, args=args, replica_id=replica_id)
            assert self.total_gpu > 0, "Megatron-Core requires at least one GPU"
            # NOTE: Only the replicas of non-trainable model will be managed by ChatLearn
            if not self.trainable:
                # NOTE: LCM(TP x CP, ETP x EP) x PP, currently only allow CP = 1
                self._num_gpu_per_replica = math.lcm(
                    self.module_args.tensor_model_parallel_size * 1,
                    self.module_args.expert_tensor_parallel_size *
                    self.module_args.expert_model_parallel_size
                ) * self.module_args.pipeline_model_parallel_size
                assert self.total_gpu % self._num_gpu_per_replica == 0, \
                    "The GPUs assigned to this model must be divisible by num_gpu_per_replica"
                self._num_replica = self.total_gpu // self._num_gpu_per_replica

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

            def set_megatron_cfg(cfg: BaseConfig):
                """
                set chatlearn cfg to megatron args
                will not set BaseConfig and key not in megatron args
                """
                for field in fields(cfg):
                    key = field.name
                    value = getattr(cfg, key)
                    if isinstance(value, BaseConfig):
                        set_megatron_cfg(value)
                    elif hasattr(args, key):
                        setattr(args, key, try_convert_to_default_type(getattr(args, key, None), value))
            set_megatron_cfg(self.module_args)

            # settings for mcore parameters micro_batch_size and global_batch_size by chatlearn args
            args.micro_batch_size = self.runtime_args.train_micro_batch_size
            args.global_batch_size = self.runtime_args.train_global_batch_size
            args.bf16 = self.module_args.bf16
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

        @property
        def megatron_args(self):
            """
            :meta private:
            """
            return get_args()

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

        def megatron_model(self):
            if isinstance(self.model, list):
                assert len(self.model) == 1
                model = self.model[0]
            else:
                model = self.model
            return model

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

        @torch.no_grad()
        def map_local_param_name_to_global(self):
            """generate a global name for each parameter in the model
            (just name of PP1EP1)
            """
            self.local_name_to_global_name = {}
            self.global_name_to_local_name = {}
            model_config = unwrap_model(self.megatron_model()).config
            # TODO: `get_transformer_layer_offset()` requires `vp_stage` in the future
            offset = get_transformer_layer_offset(model_config)
            if model_config.num_moe_experts is not None:
                ep_rank = mpu.get_expert_model_parallel_rank()
                ep_size = mpu.get_expert_model_parallel_world_size()
                num_local_experts = model_config.num_moe_experts // ep_size

            # NOTE: this regex is for model with TEGroupedGEMM
            # SequentialMLP or GroupedMLP is not supported
            regex = re.compile(r"(.*)decoder.layers\.(\d+)\.([a-z0-9_.]+)([\._])([a-z]+)([0-9]*)")
            for name, maybe_tensor in self.megatron_model().state_dict_for_save_checkpoint().items():
                if not isinstance(maybe_tensor, torch.Tensor):
                    continue
                match = regex.match(name)
                if match is None:
                    self.local_name_to_global_name[name] = name
                    self.global_name_to_local_name[name] = name
                    continue

                layer_idx = int(match.group(2)) + offset
                expert_id = ''
                if len(match.group(6)) > 0:
                    expert_id = int(match.group(6)) + num_local_experts * ep_rank
                global_name = f"{match.group(1)}decoder.layers.{layer_idx}.{match.group(3)}{match.group(4)}{match.group(5)}{expert_id}"
                self.local_name_to_global_name[name] = global_name
                self.global_name_to_local_name[global_name] = name
            return list(self.local_name_to_global_name.values())

        @torch.no_grad()
        def get_parameter_metadata(self, key_type: str='param_id'):
            """Collect parameter shape info of this rank
            """
            infos = {}
            for name, sharded_info in build_sharded_info_for_mcore_model(
                unwrap_model(self.megatron_model())
            ).items():
                param_id = self.local_name_to_param_id[name]
                sharded_info.param_id = param_id
                if key_type == 'param_id':
                    infos[param_id] = sharded_info
                elif key_type == 'local_name':
                    infos[name] = sharded_info
                else:
                    raise ValueError(f"Unsupport key_type: {key_type}")
            return infos

        def get_param_id_to_parameters(self):
            param_id_to_parameters = {}
            for name, weight in (
                unwrap_model(self.megatron_model())
                .state_dict_for_save_checkpoint()
                .items()
            ):
                if name not in self.local_name_to_param_id:
                    continue
                param_id_to_parameters[self.local_name_to_param_id[name]] = weight
            return param_id_to_parameters

else:
    class MegatronModule(TorchModule):
        """Module Placeholder for Megatron Backend"""
        # pylint: disable=abstract-method
        def __init__(self, *args, **kwargs):
            # pylint: disable=super-init-not-called
            raise SystemError("Cannot import megatron backend, please check your environment variable.")
