from chatlearn.synchronizer.scheduler import CollectiveTask, collective_task_scheduler, parallel_execute_collective_tasks
from unittest import TestCase
class TestCollectiveTaskScheduler(TestCase):
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