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

try:
    import megatron
    from megatron.core import mpu
    from megatron.initialize import initialize_megatron
    from megatron.training import save_checkpoint_and_time
except ImportError:
    print("Cannot import megatron, please set megatron python path first.")
from rlhf.utils.logger import logger
from rlhf.utils.megatron_utils import build_pipeline_layer_name_mapping
from .torch_module import RLHFTorchModule

# pylint: disable=import-outside-toplevel
class RLHFMegatronModule(RLHFTorchModule):
    """RLHFMegatronModule"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.trainable:
            # inference only
            self.model_args["micro_batch_size"] = self.module_args.generation_batch_size
        else:
            self.model_args["micro_batch_size"] = self.rlhf_args.train_micro_batch_size
            self.model_args["global_batch_size"] = self.rlhf_args.train_global_batch_size

    def add_extra_args(self, parser):
        """
        Add extra arguments for megatron.
        """
        return parser

    def init(self):
        initialize_megatron(extra_args_provider=self.add_extra_args,
                            ignore_unknown_args=True,
                            args_dict=self.model_args)

    def model_setup(self):
        """
        :meta private:
        """
        self.setup()
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
        return megatron.get_args()


    def pipeline_model_parallel_size(self):
        """
        get pipeline_model_parallel_size
        """
        return self.megatron_args.pipeline_model_parallel_size

    def tensor_model_parallel_size(self):
        """
        get tensor_model_parallel_size
        """
        return self.megatron_args.tensor_model_parallel_size

    @property
    def data_parallel_size(self):
        return mpu.get_data_parallel_world_size()

    @property
    def data_parallel_rank(self):
        return mpu.get_data_parallel_rank()

    def pipeline_parallel_rank(self):
        return mpu.get_pipeline_model_parallel_rank()

    def num_layers(self):
        """
        :meta private:
        """
        return self.megatron_args.num_layers

    def build_pipeline_layer_name_mapping(self, requires_grad=True):
        """
        :meta private:
        """
        layers_per_stage = self.num_layers() // self.pipeline_model_parallel_size()
        rank = mpu.get_pipeline_model_parallel_rank()
        logger.info(f"build mapping for rank {rank} =========")
        if isinstance(self.model, list):
            assert len(self.model) == 1
            model = self.model[0]
        else:
            model = self.model
        name_mapping = build_pipeline_layer_name_mapping(layers_per_stage, rank, model, requires_grad)
        return name_mapping

    def get_param_ranks(self):
        """
        :meta private:
        """
        # TODO: remove param_ranks in user's code
        # TODO: replace data_parallel ranks with existing methods

        param_ranks = []
        for i in range(self.data_parallel_size):
            param_ranks.append([ranks[i] for ranks in mpu.get_all_data_parallel_group_ranks()])
        self.set_param_ranks(param_ranks)
        return param_ranks

    def save_checkpoint(self, iteration):
        """
        save checkpoint at `iteration`
        :param iteration: save iteration
        """
        save_checkpoint_and_time(iteration, self.model, self.optimizer,
                                 self.opt_param_scheduler)
