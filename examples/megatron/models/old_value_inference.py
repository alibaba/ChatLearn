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
"""old value inference"""

import torch
from megatron import get_args, get_tokenizer
from megatron import print_rank_0
from megatron.core import mpu
from megatron.training import get_model
from models.value_model import ValueModel

from chatlearn import RLHFMegatronModule
from chatlearn.utils import to_device
from .constants_ppo import get_ltor_masks_and_position_ids
from .forward_step import forward_step_helper


class ValueInference(RLHFMegatronModule):
    """ValueInference"""

    def setup(self):
        self.buffer = {}
        self.stats = {}
        self.tokenizer = get_tokenizer()
        self.args = get_args()
        # Set up model and load checkpoint
        model = get_model(self.model_provider, wrap_with_ddp=False)

        assert len(model) == 1, "Above condition should have caught this"
        self.model = model[0]
        self.model.eval()
        return 'ok'

    def model_provider(self, pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building GPT model ...')
        model = ValueModel(num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process,
                           stats=self.stats, buffer=self.buffer)

        return model

    def forward_step(self, data, iteration=None):
        all_tokens = to_device("cuda", data["all_tokens"])
        output_values = None

        # =============
        # Run infernece
        # =============
        with torch.no_grad():
            attention_mask, position_ids = get_ltor_masks_and_position_ids(all_tokens)
            # logits will be meanigful only in the last pipeline stage.
            output_values = forward_step_helper(self.model, all_tokens, position_ids, attention_mask, pooling=True)

            if mpu.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert output_values is not None
        return {"old_values": output_values}
