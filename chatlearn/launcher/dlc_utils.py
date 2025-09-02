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
"""DLC utils"""

import atexit
from collections import defaultdict
import json
import os
import sys
import time
import concurrent.futures
import threading
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from chatlearn.utils import utils
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.utils.global_vars import _EXIT_ACTOR_NAME
from chatlearn.utils.log_monitor import LogMonitor, is_proc_alive, LogActor
from chatlearn.utils.utils import execute, get_ray_status

DLC_PORT_KEY = "CUSTOM_PORTS"
JOB_NAME_KEY = "JOB_NAME"
RANK_KEY = "RANK"
MASTER_ROLE = "master"
WORKER_ROLE = "worker"
PORT_SEP = ";"
LOCAL_MASTER_KEY = "LOCAL_MASTER_ADDR"
_warn_once = False
WORKER_SLEEP_SECOND = 2
_LOG_ACTOR_NAME = "CHATLEARN_LOG_ACTOR"
_EXIT_SIGNAL = False


def is_local():
    return LOCAL_MASTER_KEY in os.environ


def in_dlc_env():
    # Check whether in DLC env
    if is_local():
        # MOCK DLC in local clusters
        return True
    args = get_args()
    if not args.env_args.platform.lower() == "dlc":
        return False
    global _warn_once
    for key in [DLC_PORT_KEY, JOB_NAME_KEY, RANK_KEY]:
        if key not in os.environ:
            if not _warn_once:
                logger.warning(f"cannot find {key} in DLC env, please check whether the job is submitted in DLC "
                               f"or whether customPortList/createSvcForAllWorkers is set")
                logger.warning("fallback to local mode")
                _warn_once = True
            return False
    return True


def get_dlc_env(key):
    assert key in os.environ, f"cannot find {key} in DLC env"
    return os.environ[key]


def get_job_name():
    return get_dlc_env(JOB_NAME_KEY)


def get_master_addr():
    if is_local():
        return os.environ[LOCAL_MASTER_KEY]
    job_name = get_job_name()
    return f"{job_name}-{MASTER_ROLE}-0"


def get_rank():
    return int(get_dlc_env(RANK_KEY))


def get_addr():
    if is_local():
        return utils.get_host_addr()
    rank = get_rank()
    job_name = get_job_name()
    if rank == 0:
        role = MASTER_ROLE
        index = 0
    else:
        role = WORKER_ROLE
        index = rank - 1
    return f"{job_name}-{role}-{index}"


def get_free_ports():
    # port for DLC jobs
    assert DLC_PORT_KEY in os.environ, f"cannot find port {DLC_PORT_KEY} in DLC"
    free_ports = [int(port) for port in os.environ[DLC_PORT_KEY].strip().split(PORT_SEP)]

    # remove ports that reserved by ray
    # 'client_server': 10001, 'dashboard': 8265, 'dashboard_agent_grpc': 49948, 'dashboard_agent_http': 52365,
    # 'metrics_export': 63529, 'redis_shards': 'random', 'worker_ports': '9998 ports from 10002 to 19999'
    def _valid_port(port):
        if port in [10001, 8265, 49948, 52365, 63529]:
            return False
        if 10002 <= port <= 19999:
            return False
        return True

    free_ports = [port for port in free_ports if _valid_port(port)]
    return free_ports


def start_ray_cluster():
    free_ports = get_free_ports()
    port = free_ports[0]
    node_manager_port = free_ports[1]
    master_addr = get_master_addr()
    rank = get_rank()
    system_config = json.dumps({"object_timeout_milliseconds": 30000})
    if rank == 0:
        cmd = f"RAY_prestart_worker_first_driver=0 ray start --head --port={port} --node-ip-address={master_addr} " + \
              f"--node-manager-port {node_manager_port} --node-name={master_addr} --system-config='{system_config}' " + \
              "--dashboard-host=0.0.0.0 --dashboard-port=8265"
    else:
        cmd = f"ray start --address={master_addr}:{port} --node-manager-port {node_manager_port} " + \
              f"--node-name={get_addr()} --dashboard-host=0.0.0.0 --dashboard-port=8265"
    logger.info(f"execute {cmd}")
    state, _ = execute(cmd)
    if not state:
        sys.exit(1)

def filter_known_msg(msg):
    if "StatusCode.DEADLINE_EXCEEDED" in msg:
        return True
    return False


@ray.remote
class ExitActor:
    """ExitActor"""

    def __init__(self):
        self._node_and_err_msg = defaultdict(list)

    def notify(self):
        return 1

    def add_error_node_and_msg(self, ip, msg):
        self._node_and_err_msg[ip].append(msg)

    def get_error_node_and_msg(self):
        return self._node_and_err_msg

    def get_error_msg(self, ip):
        return self._node_and_err_msg[ip]

def execute_with_timeout(func, args, timeout):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args)
        try:
            result = future.result(timeout)
            return result
        except concurrent.futures.TimeoutError:
            future.cancel()
            print("Function execution timed out.")
        except Exception:
            # actor has not been created yet
            return

class StartExitListener:
    """StartExitListener"""

    def __init__(self):
        log_dir = os.path.dirname(os.path.dirname(ray.nodes()[0]['ObjectStoreSocketName']))
        self.log_dir = os.path.join(log_dir, 'logs')
        print(self.log_dir, flush=True)
        log_actor = None
        # Only run the actor on the master node.
        if get_rank() == 0:
            log_actor = LogActor.options(
                name=_LOG_ACTOR_NAME,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft = False,
                ), lifetime="detached"
            ).remote()
        else:
            while log_actor is None:
                try:
                    log_actor = ray.get_actor(_LOG_ACTOR_NAME)
                except Exception:
                    print(f'get actor {_LOG_ACTOR_NAME} failed, retry ....')
                    time.sleep(2)

        self.log_monitor = LogMonitor(
            self.log_dir,
            is_proc_alive,
            log_actor
        )
        self._start_exit_actor = None
        self.quit_event = threading.Event()
        self.log_monitor_thread = threading.Thread(target=self.log_monitor.run, args=(self.quit_event,))
        self.log_monitor_thread.daemon = True
        self.log_monitor_thread.start()

    def stop(self):
        self.quit_event.set()
        self.log_monitor_thread.join(2)
        ray.shutdown()
        logger.info("Execute ray.shutdown before the program exits. Done ...")

    def start_exit_listener(self):
        atexit.register(self.stop)
        address = get_addr()
        if get_rank() == 0:
            self._start_exit_actor = ExitActor.options(name=_EXIT_ACTOR_NAME, lifetime="detached").remote()
        else:
            # wait for the head node to be created
            head_created = False
            while True:
                cluster_state, msg = get_ray_status()
                if cluster_state:
                    if msg is None:
                        head_created = True
                    else:
                        if not filter_known_msg(msg):
                            logger.warning(f"ray status got unknown msg {msg}, ignore ...")
                else:
                    if head_created:
                        logger.info(f"ray status got msg {msg}")
                        logger.info("head has exited, exit worker ...")
                        break
                    logger.info("wait for head to be created.")
                if self._start_exit_actor is None:
                    self._start_exit_actor = execute_with_timeout(ray.get_actor, [_EXIT_ACTOR_NAME], 3)
                if self._start_exit_actor is not None:
                    try:
                        error_msg_list = ray.get(self._start_exit_actor.get_error_msg.remote(address))
                    except ray.exceptions.RayActorError:
                        logger.info("start_exit_actor has been killed")
                        break
                    if error_msg_list:
                        msg = '\n'.join(error_msg_list)
                        raise Exception(msg)
                time.sleep(WORKER_SLEEP_SECOND)
            print("Exit worker", flush=True)
