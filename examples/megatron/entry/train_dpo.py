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

from examples.megatron.models import PolicyReference, PolicyTrainer
from examples.megatron.models.train_helper import get_prompts
import chatlearn
from chatlearn import DPOEngine


if __name__ == "__main__":
    chatlearn.init()
    args = chatlearn.get_args()
    reference_model = PolicyReference("reference")
    policy_trainer = PolicyTrainer("ppo_policy")

    engine = DPOEngine(reference_model, policy_trainer)

    all_prompts = get_prompts(args.runtime_args.data_path, num_limit=args.runtime_args._args_dict['training_data_num_limit'])
    random.seed(reference_model.model_args["seed"])
    num_train = len(all_prompts)
    random.shuffle(all_prompts)
    train_prompts = all_prompts[:num_train]

    engine.set_dataset(train_prompts)
    engine.learn()
