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
"""UT for sampler."""

import unittest

from itertools import chain
from chatlearn.data.sampler import EpisodeDataSampler


# pylint: disable=missing-class-docstring
class TestDataset(unittest.TestCase):

    def test_circle_episode_small_dataset(self):
        num_replicas = 2
        samplers = [EpisodeDataSampler(
            total_samples=3,
            consumed_samples=0,
            micro_batch_size=8,
            data_parallel_rank=i,
            data_parallel_size=num_replicas,
            sample_per_episode=32
        ) for i in range(num_replicas)]

        res = [[] for _ in range(num_replicas)]
        for idx, sampler in enumerate(samplers):
            for indices in sampler:
                res[idx].append(indices)
                if len(res[idx]) > 6:
                    break
        expect_0 = [
            [0, 1, 2, 0, 1, 2, 0, 1],
            [1, 2, 0, 1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0, 1, 2, 0],
            [0, 1, 2, 0, 1, 2, 0, 1],
            [1, 2, 0, 1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0, 1, 2, 0],
            [0, 1, 2, 0, 1, 2, 0, 1]
        ]

        expect_1 = [
            [2, 0, 1, 2, 0, 1, 2, 0],
            [0, 1, 2, 0, 1, 2, 0, 1],
            [1, 2, 0, 1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0, 1, 2, 0],
            [0, 1, 2, 0, 1, 2, 0, 1],
            [1, 2, 0, 1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0, 1, 2, 0]
        ]

        self.assertEqual(res, [expect_0, expect_1])


    def test_circle_episode_data(self):
        num_replicas = 2
        samplers = [EpisodeDataSampler(
            total_samples=25,
            consumed_samples=0,
            micro_batch_size=8,
            data_parallel_rank=i,
            data_parallel_size=num_replicas,
            sample_per_episode=32
        ) for i in range(num_replicas)]

        res = [[] for _ in range(num_replicas)]
        for idx, sampler in enumerate(samplers):
            for indices in sampler:
                res[idx].append(indices)
                if len(res[idx]) > 6:
                    break
        expect_0 = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [7, 8, 9, 10, 11, 12, 13, 14], 
            [23, 24, 0, 1, 2, 3, 4, 5], 
            [14, 15, 16, 17, 18, 19, 20, 21], 
            [5, 6, 7, 8, 9, 10, 11, 12], 
            [21, 22, 23, 24, 0, 1, 2, 3]
        ]
        expect_1 = [
            [8, 9, 10, 11, 12, 13, 14, 15],
            [24, 0, 1, 2, 3, 4, 5, 6],
            [15, 16, 17, 18, 19, 20, 21, 22],
            [6, 7, 8, 9, 10, 11, 12, 13],
            [22, 23, 24, 0, 1, 2, 3, 4],
            [13, 14, 15, 16, 17, 18, 19, 20],
            [4, 5, 6, 7, 8, 9, 10, 11]
        ]
        for idx, ele in enumerate(res):
            print(f"res_{idx}: {ele}")

        self.assertEqual(res, [expect_0, expect_1])

    def test_circle_episode_data_drop_last(self):
        num_replicas = 2
        samplers = [EpisodeDataSampler(
            total_samples=35,
            consumed_samples=0,
            micro_batch_size=8,
            data_parallel_rank=i,
            data_parallel_size=num_replicas,
            sample_per_episode=32,
            drop_last=True
        ) for i in range(num_replicas)]

        res = [[] for _ in range(num_replicas)]
        for idx, sampler in enumerate(samplers):
            for indices in sampler:
                res[idx].append(indices)
                if len(res[idx]) > 6:
                    break
        expect_0 = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 6, 7]
        ]
        expect_1 = [
            [8, 9, 10, 11, 12, 13, 14, 15],
            [24, 25, 26, 27, 28, 29, 30, 31],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [24, 25, 26, 27, 28, 29, 30, 31],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [24, 25, 26, 27, 28, 29, 30, 31],
            [8, 9, 10, 11, 12, 13, 14, 15]
        ]
        for idx, ele in enumerate(res):
            print(f"res_{idx}: {ele}")

        self.assertEqual(res, [expect_0, expect_1])

# pylint: enable=missing-class-docstring


if __name__ == '__main__':
    unittest.main()
