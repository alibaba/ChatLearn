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
"""UnitTest for CollectiveTaskScheduler ."""

from unittest import TestCase
from chatlearn.synchronizer.scheduler import CollectiveTask, collective_task_scheduler, parallel_execute_collective_tasks


class TestCollectiveTaskScheduler(TestCase):
    """Test for CollectiveTaskScheduler"""
    def test_scheduler(self):
        tasks = [
            CollectiveTask([1, 2], "group1"),
            CollectiveTask([1, 3], "group1"),
            CollectiveTask([2, 1], "group2"),
            CollectiveTask([4, 5], "group2"),
        ]
        generator = collective_task_scheduler(tasks)
        paralel_tasks = next(generator)
        self.assertEqual([1, 2], paralel_tasks[0].actors)
        self.assertEqual([4, 5], paralel_tasks[1].actors)
        paralel_tasks = next(generator)
        self.assertEqual([1, 3], paralel_tasks[0].actors)
        paralel_tasks = next(generator)
        self.assertEqual([2, 1], paralel_tasks[0].actors)

        def task_func(task):
            if task.actors[0] == 2:
                self.assertEqual("group2", task.group)

        parallel_execute_collective_tasks(tasks, task_func)
