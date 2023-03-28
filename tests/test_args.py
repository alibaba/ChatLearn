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
"""UT for args."""

from rlhf.arguments import parse_args


def test_args():
    args0 = parse_args()
    assert args0.models["policy"].model_config_file == "model.yaml"
    assert args0.rlhf_args.num_training_epoch == 3
    assert args0.models['reference'].gpu_per_process == 1
    assert args0.models['policy'].model_args['generate_config']["num_beams"] == 1
    assert args0.rlhf_args.get("unknown_args") == "test_unknown"
    assert args0.models['policy'].model_args['model_config']['attention_probs_dropout_prob'] == 0.1
    assert args0.models['policy'].model_args['test'] == 123


if __name__ == '__main__':
    test_args()
