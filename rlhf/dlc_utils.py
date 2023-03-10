import os
import subprocess
import time
import ray
from rlhf.global_vars import set_exit_actor


DLC_PORT_KEY = "CUSTOM_PORTS"
JOB_NAME_KEY = "JOB_NAME"
RANK_KEY = "RANK"
MASTER_ROLE = "master"
WORKER_ROLE = "worker"
PORT_SEP = ";"


def get_dlc_env(key):
    assert key in os.environ, f"cannot find {key} in DLC env"
    return os.environ[key]


def get_job_name():
    return get_dlc_env(JOB_NAME_KEY)


def get_master_addr():
    job_name = get_job_name()
    return f"{job_name}-{MASTER_ROLE}-0"


def get_rank():
    return int(get_dlc_env(RANK_KEY))


def get_addr():
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
    return free_ports


def start_ray_cluster():
    job_name = get_job_name()
    port = get_free_ports()[0]
    master_addr = get_master_addr()
    rank = get_rank()
    if rank == 0:
        cmd = f"ray start --head --port={port} --node-ip-address={master_addr}"
    else:
        cmd = f"ray start --address={master_addr}:{port}"
    print(f"execute {cmd}")
    subprocess.run(cmd, shell=True)


@ray.remote
class ExitActor:

    def notify(self):
        return 1


def start_exit_listener():
    name = "ExitActor"
    if get_rank() == 0:
        actor = ExitActor.options(name=name).remote()
        # avoid actor GC
        set_exit_actor(actor)
    else:
        # wait for the head node to create ExitActor
        head_created = False
        while True:
            try:
                ray.get_actor(name)
                head_created = True
            except ValueError:
                if head_created:
                    return
            time.sleep(5)
