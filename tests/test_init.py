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
"""UT for init."""

import unittest
import torch

from rlhf.model.base_model import BaseModel
from rlhf.initialize import init_process_group

# pylint: disable=missing-docstring
class InitTest(unittest.TestCase):

    def test_init(self):
        policy = BaseModel(4, 'policy')
        value = BaseModel(4, 'value')
        models = [policy, value]
        init_process_group(models, './')
        self.assertTrue(torch.distributed.is_initialized())
        if policy.global_ranks is not None:
            self.assertEqual(policy.global_ranks, [0, 1, 2, 3])
        else:
            self.assertEqual(value.global_ranks, [4, 5, 6, 7])
# pylint: enable=missing-docstring


if __name__ == '__main__':
    unittest.main()
