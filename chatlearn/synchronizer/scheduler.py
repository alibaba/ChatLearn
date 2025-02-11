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
"""
CollectiveTaskScheduler uses two queue to schedule collective task:
- TodoQueue maintenances all tasks that to be executed
- PendingQueue maintenances execution tasks

This scheduler resort the remote tasks to avoid collective operator hang.
"""
from queue import Queue
import concurrent
import traceback
from concurrent.futures import ThreadPoolExecutor

from chatlearn.utils.logger import logger

class CollectiveTask:
    """ColleciteTask represents a group of actors to execute a collective task"""
    def __init__(self, actors, group):
        self.actors = actors
        self.group = group

def collective_task_scheduler(tasks):

    todo_queue = Queue()
    pending_queue = []
    _ = [todo_queue.put(task) for task in tasks]

    while not todo_queue.empty():
        send_actors_set = set()
        recv_actors_set = set()
        list_count = todo_queue.qsize()
        # re-put it if confilict, otherwise put it to PendingQueue
        for _ in range(list_count):
            task = todo_queue.get()
            send = task.actors[0]
            recvs = task.actors[1:]
            if send not in send_actors_set and send not in recv_actors_set and \
                all(recv not in send_actors_set for recv in recvs) and all(recv not in recv_actors_set for recv in recvs):
                pending_queue.append(task)
                send_actors_set.add(send)
                recv_actors_set.update(recvs)
            else:
                todo_queue.put(task)
        if pending_queue:
            yield pending_queue
            send_actors_set = set()
            recv_actors_set = set()
            pending_queue = []

def parallel_execute_collective_tasks(tasks, submit_func):
    scheduler = collective_task_scheduler(tasks)
    for parallel_tasks in scheduler:
        logger.info(f"DEBUG parallel_execute_tasks: {[task.group for task in parallel_tasks]}")
        with ThreadPoolExecutor(max_workers=len(parallel_tasks)) as executor:
            futures = [executor.submit(submit_func, task) for task in parallel_tasks]
            for _future in concurrent.futures.as_completed(futures):
                try:
                    _future.result()
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"ParameterSync warmup failed: {e}") # pylint: disable=raise-missing-from
