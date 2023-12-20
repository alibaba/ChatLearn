# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
"""RLHF Megatron module"""

import inspect
import torch

try:
    import megatron
    from megatron.core import mpu
    from megatron.initialize import set_jit_fusion_options
    from megatron.training import save_checkpoint_and_time
    from megatron.initialize import initialize_megatron
    from chatlearn.utils.megatron_utils import initialize_megatron as chatlearn_initialize_megatron
    from chatlearn.utils.megatron_utils import build_pipeline_layer_name_mapping
except ImportError:
    print("Cannot import megatron, please set megatron python path first.")
from .torch_module import RLHFTorchModule

# pylint: disable=import-outside-toplevel
class RLHFMegatronModule(RLHFTorchModule):
    """RLHFMegatronModule is the class for RLHF Megatron models.

    Args
    ----
    name : str
        model name
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.trainable:
            # inference only
            if self.model_args.get("micro_batch_size") != self.module_args.generation_batch_size:
                self._logger.info(f"{self.name} Overwrite micro_batch_size with generation_batch_size {self.module_args.generation_batch_size}")
            self.model_args["micro_batch_size"] = self.module_args.generation_batch_size
        else:
            self.model_args["micro_batch_size"] = self.rlhf_args.train_micro_batch_size
            self.model_args["global_batch_size"] = self.rlhf_args.train_global_batch_size
            if self.model_args.get("micro_batch_size") != self.rlhf_args.train_micro_batch_size:
                self._logger.info(f"{self.name} Overwrite micro_batch_size with train_micro_batch_size {self.module_args.train_micro_batch_size}")
            if self.model_args.get("global_batch_size") != self.rlhf_args.train_global_batch_size:
                self._logger.info(f"{self.name} Overwrite global_batch_size with train_global_batch_size {self.module_args.train_global_batch_size}")

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
        else:
            assert hasattr(self, "model")
            self.model.eval()

    @property
    def megatron_args(self):
        """
        :meta private:
        """
        return megatron.get_args()

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
        if isinstance(self.model, list):
            assert len(self.model) == 1
            model = self.model[0]
        else:
            model = self.model
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
        if self.to_offload_optimizer_states:
            timer = self.timers("offload")
            if not timer.started_:
                timer.start()
            # offload onto cpu
            self._optimizer_load_state_bucket_into_device(device='cpu')
            self.empty_cache()
            timer.stop()

    def onload_optimizer_states(self):
        """
        onload optimizer states
        """
        if self.to_offload_optimizer_states:
            timer = self.timers("onload")
            if not timer.started_:
                timer.start()
            self._optimizer_load_state_bucket_into_device(device=torch.cuda.current_device())
            timer.stop()

    def _optimizer_load_state_bucket_into_device(self, device):
        """put the state bucket onto a device
        """
        state_dict = self.optimizer.optimizer.state_dict()
        for tensors in state_dict['state'].values():
            keys = list(tensors.keys())
            for key in keys:
                tensors[key] = tensors[key].to(device=device, non_blocking=True)
        # make sure the loading is finished before returning
        torch.cuda.synchronize()
