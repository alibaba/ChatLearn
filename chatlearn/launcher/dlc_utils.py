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
"""DLC utils"""

import os
import subprocess
import time

from chatlearn.utils import utils
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger

DLC_PORT_KEY = "CUSTOM_PORTS"
JOB_NAME_KEY = "JOB_NAME"
RANK_KEY = "RANK"
MASTER_ROLE = "master"
WORKER_ROLE = "worker"
PORT_SEP = ";"
LOCAL_MASTER_KEY = "LOCAL_MASTER_ADDR"
_warn_once = False


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


def execute(cmd, check=False, retry=1):
    """
    Execute cmd in shell
    
    Args:
        check: if returncode is non-zero, raise error
    """
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
    state = ret.returncode == 0
    msg = ret.stdout if state else ret.stderr
    if not state and retry > 1:
        logger.warning(f"execute {cmd} got error {msg}, retry...")
        time.sleep(1)
        return execute(cmd, check, retry-1)
    return state, msg


def start_ray_cluster():
    port = get_free_ports()[0]
    master_addr = get_master_addr()
    rank = get_rank()
    if rank == 0:
        cmd = f"ray start --head --port={port} --node-ip-address={master_addr}"
    else:
        cmd = f"ray start --address={master_addr}:{port}"
    logger.info(f"execute {cmd}")
    execute(cmd, check=True)


def start_exit_listener():
    if get_rank() != 0:
        # wait for the head node to be created
        head_created = False
        counter = 0
        while True:
            cluster_state, msg = execute("ray status", retry=3)
            if cluster_state:
                head_created = True
                # log per one hour
                if counter % 720 == 0:
                    logger.info("worker is listening to head")
                    logger.info(msg)
                counter += 1
            elif "StatusCode.UNAVAILABLE" in msg and "Connection refused" in msg:
                if head_created:
                    logger.info(f"ray status got error {msg}")
                    logger.info("head has exited, exit worker ...")
                    execute("ray stop", check=True)
                    return
                else:
                    logger.info("wait for head to be created.")
            else:
                logger.warning(f"ray status got error {msg}")
            time.sleep(5)
