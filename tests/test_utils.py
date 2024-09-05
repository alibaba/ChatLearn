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
"""UT for utils."""

import unittest
import ray
from chatlearn.utils.utils import parse_function_return_num
from chatlearn import get
from chatlearn.utils.utils import split_index


# pylint: disable=missing-class-docstring
class TestDataset(unittest.TestCase):

    def test_function_return(self):

        def func():
            return 1, 2, 3

        res = parse_function_return_num(func)
        self.assertEqual(res, 3)
        
        def func2(aaa):
            if aaa > 0:
                return 1, 2
            else:
                return 3, 4

        res = parse_function_return_num(func2)
        self.assertEqual(res, 2)
        
        def func3():
            res = [1, 2, 3]
            return res

        res = parse_function_return_num(func3)
        self.assertEqual(res, 1)
        
        def func4():
            res = [1, 2, 3]
            return res, 1

        res = parse_function_return_num(func4)
        self.assertEqual(res, 2)

    def _test_get(self):
        ray.init()
        value = ray.put(1)
        data = (value, {1:1})
        data1 = get(data)
        self.assertEqual(data1, (1, {1:1}))
        data = ([value], {1:1})
        data1 = get(data)
        self.assertEqual(data1, ([1], {1:1}))
        value = ray.put({"a":2})
        data = ([value], {1:1})
        data1 = get(data)
        self.assertEqual(data1, ([{"a":2}], {1:1}))

    def test_split_index(self):
        length = 10
        num_splits = 3
        res = split_index(length,num_splits)
        self.assertEqual(res, [(0, 4), (4, 7), (7, 10)])

# pylint: enable=missing-class-docstring


if __name__ == '__main__':
    unittest.main()
