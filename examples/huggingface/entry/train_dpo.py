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
"""entry file for training dpo"""

import random

from examples.huggingface.models.dpo.policy_trainer import PolicyTrainer
from examples.huggingface.models.dpo.reference_model import ReferenceModel

import chatlearn
from chatlearn.models.deepspeed.deepspeed_utils import get_tokenizer
from chatlearn import DPOEngine
from models.utils import blending_datasets


if __name__ == "__main__":
    chatlearn.init()
    args = chatlearn.get_args()
    reference = ReferenceModel("reference")
    policy = PolicyTrainer("policy_trainer")

    engine = DPOEngine(reference, policy)

    # prepare datasets
    prompts_data = blending_datasets(
        reference.model_args['reward_data'],
        reference.model_args['reward_data_probs'],
        reference.model_args['seed'],
        return_eval=False,
    )
    engine.set_dataset(prompts_data)
    engine.learn()
