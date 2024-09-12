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
"""Megatron module"""
import inspect

try:
    from chatlearn.utils.megatron_import_helper import get_args
    from chatlearn.utils.megatron_import_helper import mpu
    from chatlearn.utils.megatron_import_helper import initialize_megatron
    from chatlearn.utils.megatron_import_helper import save_checkpoint_and_time
    from chatlearn.utils.megatron_import_helper import set_jit_fusion_options
    from chatlearn.utils.megatron_utils import initialize_megatron as chatlearn_initialize_megatron
    from chatlearn.utils.megatron_utils import build_pipeline_layer_name_mapping
    from chatlearn.models.megatron.memory_manager import create_trainer_memory_manager, InferenceMemoryManager
except ImportError:
    mpu = None
from .torch_module import TorchModule


# pylint: disable=import-outside-toplevel
class MegatronModule(TorchModule):
    """MegatronModule is the class for Alignment Megatron models.

    Args
    ----
    name : str
        model name
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mpu is None:
            print("Cannot import megatron, please set megatron python path first.")
        if not self.trainable:
            # inference only
            if self.model_args.get("micro_batch_size") != self.module_args.generation_batch_size:
                self._logger.info(f"{self.name} Overwrite micro_batch_size with generation_batch_size {self.module_args.generation_batch_size}")
            self.model_args["micro_batch_size"] = self.module_args.generation_batch_size
        else:
            self.model_args["micro_batch_size"] = self.runtime_args.train_micro_batch_size
            self.model_args["global_batch_size"] = self.runtime_args.train_global_batch_size
            if self.model_args.get("micro_batch_size") != self.runtime_args.train_micro_batch_size:
                self._logger.info(f"{self.name} Overwrite micro_batch_size with train_micro_batch_size {self.module_args.train_micro_batch_size}")
            if self.model_args.get("global_batch_size") != self.runtime_args.train_global_batch_size:
                self._logger.info(f"{self.name} Overwrite global_batch_size with train_global_batch_size {self.module_args.train_global_batch_size}")
        if not self.model_args.get("tensorboard_dir") and self.runtime_args.output_dir is not None:
            self.model_args['tensorboard_dir'] = f"{self.runtime_args.output_dir}/tensorboard"


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
        if "args_dict" in inspect.getfullargspec(initialize_megatron).args:
            initialize_func = initialize_megatron
        else:
            initialize_func = chatlearn_initialize_megatron
        initialize_func(extra_args_provider=self.add_extra_args,
                        ignore_unknown_args=True,
                        args_dict=self.model_args)
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
            if self.module_args.offload_weights or self.module_args.free_grad_buffers or self.module_args.offload_optimizer_states:
                self._memory_manager = create_trainer_memory_manager(
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
            self.model.eval()
            if self.module_args.offload_weights:
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

    def build_pipeline_layer_name_mapping(self, num_target_pipe_stage, target_pipe_rank, requires_grad=True):
        """
        build name mapping from src model to tgt model
        Args:
            num_target_pipe_stage: number of pipeline stage in target model
            target_pipe_rank: target model pipeline rank
            requires_grad: whether the returned layer requires_grad, as we only need to sync parameters that have changed

        :meta private:
        """
        src_layers_per_stage = self.num_layers() // self.pipeline_model_parallel_size()
        dst_layers_per_stage = self.num_layers() // num_target_pipe_stage
        assert dst_layers_per_stage % src_layers_per_stage == 0, \
            "We assume pipeline stage of target model is smaller than src model, and is divisible by src model"
        mapping_interval = dst_layers_per_stage // src_layers_per_stage
        src_rank = mpu.get_pipeline_model_parallel_rank()
        self._logger.debug(f"build mapping for rank {src_rank} =========")
        model = self.megatron_model()
        is_tgt_last_stage = target_pipe_rank == num_target_pipe_stage - 1 and target_pipe_rank != 0
        name_mapping = build_pipeline_layer_name_mapping(src_layers_per_stage, src_rank, mapping_interval,
                                                         is_tgt_last_stage, model, requires_grad)
        return name_mapping

    def get_local_param_ranks(self):
        """
        :meta private:
        """
        data_parallel_global_ranks = list(mpu._DATA_PARALLEL_GLOBAL_RANKS)
        return data_parallel_global_ranks, mpu.get_data_parallel_rank()

    def save_checkpoint(self, iteration):
        """
        save checkpoint at `iteration`
        :param iteration: save iteration
        
        :meta private:
        """
        if self.enable_lora:
            self.fuse_lora_layer()
        save_checkpoint_and_time(iteration, self.model, self.optimizer,
                                 self.opt_param_scheduler)
        if self.enable_lora:
            self.unfuse_lora_layer()

    def offload_optimizer_states(self):
        """
        offload optimizer states
        """
        if self.module_args.offload_optimizer_states:
            self._memory_manager.offload_optimizer_states()

    def onload_optimizer_states(self):
        """
        onload optimizer states
        """
        if self.module_args.offload_optimizer_states:
            self._memory_manager.onload_optimizer_states()

    def offload_main_weights(self):
        """
        offload main weights
        """
        if self.module_args.offload_weights:
            self._memory_manager.offload_main_weights()

    def onload_main_weights(self):
        """
        onload main weights
        """
        if self.module_args.offload_weights:
            self._memory_manager.onload_main_weights()

    def offload_weights(self):
        """
        offload weights
        """
        if self.module_args.offload_weights:
            self._memory_manager.offload_weights()

    def onload_weights(self):
        """
        onload weights
        """
        if self.module_args.offload_weights:
            self._memory_manager.onload_weights()

    def free_grad_buffers(self):
        """
        free grad buffers and related tensors
        """
        if self.module_args.free_grad_buffers:
            self._memory_manager.free_grad_buffers()

    def build_grad_buffers(self):
        """
        build grad buffers and related tensors
        """
        if self.module_args.free_grad_buffers:
            self._memory_manager.build_grad_buffers()
