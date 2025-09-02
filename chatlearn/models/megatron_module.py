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
from dataclasses import fields

import inspect
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

from chatlearn.configs import BaseConfig

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
    from chatlearn.utils.mappings.megatron_helpers import build_sharded_info_for_mcore_model

    # pylint: disable-next=ungrouped-imports
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    class MegatronModule(TorchModule):
        """MegatronModule is the class for Alignment Megatron models."""
        # pylint: disable=abstract-method
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
            args.train_iters = 1

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

            def set_megatron_cfg(cfg: BaseConfig, used_names: set=None):
                """
                set chatlearn cfg to megatron args
                will not set BaseConfig and key not in megatron args
                """
                if used_names is None:
                    used_names = set()
                for field in fields(cfg):
                    key = field.name
                    value = getattr(cfg, key)
                    if isinstance(value, BaseConfig):
                        set_megatron_cfg(value, used_names)
                    elif hasattr(args, key):
                        if key in used_names:
                            raise ValueError(f"Attempt to pass {key} to Megatron twice")
                        used_names.add(key)
                        setattr(args, key, try_convert_to_default_type(getattr(args, key, None), value))
            set_megatron_cfg(self.module_args)

            # settings for mcore parameters micro_batch_size and global_batch_size by chatlearn args
            args.micro_batch_size = self.runtime_args.train_micro_batch_size
            args.global_batch_size = self.runtime_args.train_global_batch_size
            args.bf16 = self.module_args.bf16
            initialize_megatron(parsed_args=args)

            # NOTE: Megatron-Core will override variable_seq_lengths to be False, override it back
            get_args().variable_seq_lengths = self.module_args.variable_seq_lengths

            self.num_train_global_batch = self.runtime_args.sample_per_episode // self.runtime_args.train_global_batch_size

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
                        self.model,
                        self.optimizer,
                        self.runtime_args.bucket_size_mb_in_memory_manager,
                    )
                    self.offload()
            else:
                assert hasattr(self, "model")
                for model_chunk in self.model:
                    model_chunk.eval()
                if self.module_args.free_gpu_memory.offload_weights:
                    self._memory_manager = InferenceMemoryManager(
                        self.model,
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

        def save_checkpoint(self, iteration):
            """
            save checkpoint at `iteration`
            :param iteration: save iteration

            :meta private:
            """
            get_args().save = f"{self.runtime_args.output_dir}/save_model/{self.name}"
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
            """Map names of weights on each rank to a unique name.
            
            Returns:
                List[str]: A list of unique global names for each weight 
            on this rank.
            """
            self.global_name_to_local_name = {}
            # NOTE: this regex is for model with TEGroupedGEMM
            # SequentialMLP or GroupedMLP is not supported
            regex = re.compile(r"(.*)decoder.layers\.(\d+)\.([a-z0-9_.]+)([\._])([a-z]+)([0-9]*)")
            for vp_stage, model_chunk in enumerate(self.model):
                model_config = unwrap_model(model_chunk).config
                if 'vp_stage' in inspect.signature(get_transformer_layer_offset).parameters:
                    offset = get_transformer_layer_offset(model_config, vp_stage=vp_stage)
                else:
                    if len(self.model) > 1:
                        mpu.set_virtual_pipeline_model_parallel_rank(vp_stage)
                    offset = get_transformer_layer_offset(model_config)
                    if len(self.model) > 1:
                        mpu.set_virtual_pipeline_model_parallel_rank(None)
                if model_config.num_moe_experts is not None:
                    ep_rank = mpu.get_expert_model_parallel_rank()
                    ep_size = mpu.get_expert_model_parallel_world_size()
                    num_local_experts = model_config.num_moe_experts // ep_size

                for name, maybe_tensor in model_chunk.state_dict_for_save_checkpoint().items():
                    if not isinstance(maybe_tensor, torch.Tensor):
                        continue
                    match = regex.match(name)
                    local_name = f"{vp_stage}-{name}"
                    if match is None:
                        self.global_name_to_local_name[name] = local_name
                        continue

                    layer_idx = int(match.group(2)) + offset
                    expert_id = ''
                    if len(match.group(6)) > 0:
                        expert_id = int(match.group(6)) + num_local_experts * ep_rank
                    global_name = f"{match.group(1)}decoder.layers.{layer_idx}.{match.group(3)}{match.group(4)}{match.group(5)}{expert_id}"
                    self.global_name_to_local_name[global_name] = local_name
            return list(self.global_name_to_local_name.keys())

        @torch.no_grad()
        def get_parameter_metadata(self, key_type: str='param_id'):
            """Collect parameter shape info of this rank
            """
            infos = {}
            # NOTE: encode local_name: "{vp_stage}-{weight_name}"
            for vp_stage, model_chunk in enumerate(self.model):
                for name, sharded_info in build_sharded_info_for_mcore_model(
                    unwrap_model(model_chunk)
                ).items():
                    local_name = f"{vp_stage}-{name}"
                    param_id = self.local_name_to_param_id[local_name]
                    sharded_info.param_id = param_id
                    if key_type == 'param_id':
                        infos[param_id] = sharded_info
                    elif key_type == 'local_name':
                        infos[local_name] = sharded_info
                    else:
                        raise ValueError(f"Unsupport key_type: {key_type}")
            return infos

        def parameter_sync(self):
            """Perform parameter synchronization on this worker."""
            if self.synchronizer is None:
                raise ValueError("Synchronizer is not initialized.")
            param_id_to_parameters = {}
            for vp_stage, model_chunk in enumerate(self.model):
                for name, weight in (
                    unwrap_model(model_chunk)
                    .state_dict_for_save_checkpoint()
                    .items()
                ):
                    local_name = f"{vp_stage}-{name}"
                    if local_name not in self.local_name_to_param_id:
                        continue
                    param_id_to_parameters[self.local_name_to_param_id[local_name]] = weight

            self.param_id_to_parameters = param_id_to_parameters
            self.synchronizer.parameter_sync()

        def post_parameter_sync(self):
            self.param_id_to_parameters = None


else:
    class MegatronModule(TorchModule):
        """Module Placeholder for Megatron Backend"""
        # pylint: disable=abstract-method
        def __init__(self, *args, **kwargs):
            # pylint: disable=super-init-not-called
            raise SystemError("Cannot import megatron backend, please check your environment variable.")
