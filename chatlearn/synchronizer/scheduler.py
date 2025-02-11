
from queue import Queue
import concurrent
import traceback
from chatlearn.utils.logger import logger
from concurrent.futures import ThreadPoolExecutor

class CollectiveTask:
    def __init__(self, actors, group):
        self.actors = actors
        self.group = group
def collective_task_scheduler(tasks):
    # using two queue to schedule collective task:
    # - TodoQueue maintenances all tasks that to be executed
    # - PendingQueue maintenances execution tasks
    # to avoid hang in Ray RemoteActor, this scheduler do not want
    # one actor has differnt role at one time(send and recv)
    todo_queue = Queue()
    pending_queue = []
    [todo_queue.put(task) for task in tasks]
    
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
