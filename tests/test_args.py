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
"""UT for args."""

import os

from chatlearn.utils.arguments import parse_args


def test_args():
    args0 = parse_args()
    assert args0.runtime_args.num_training_epoch == 3, args0.runtime_args.num_training_epoch
    assert args0.models["policy"].model_config_file == "configs/model.yaml", args0.models["policy"].model_config_file
    assert args0.models['reference'].gpu_per_process == 1
    assert args0.models['policy'].args_dict['generate_config']["num_beams"] == 1
    assert args0.runtime_args.get("unknown_args") == "test_unknown"
    assert args0.models['policy'].args_dict['model_config']['attention_probs_dropout_prob'] == 0.1
    assert args0.models['policy'].args_dict['test'] == 123
    assert args0.models['policy'].args_dict['generate_config']['eos_token_id'] == 103

def test_args2():
    os.environ["num_training_epoch"] = "2"
    args0 = parse_args()
    assert args0.runtime_args.num_training_epoch == 2


if __name__ == '__main__':
    test_args()
    test_args2()
