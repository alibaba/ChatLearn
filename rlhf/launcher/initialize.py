import os
import sys

import ray
import torch
from cupy.cuda import nccl
from ray.util.collective.collective_group.nccl_util import TORCH_NCCL_DTYPE_MAP

from rlhf.launcher import dlc_utils
from rlhf.utils.arguments import parse_args
from rlhf.utils.global_vars import set_global_variables
from rlhf.utils.global_vars import set_initialized
from rlhf.utils.logger import logger
from rlhf.utils.version import VERSION


# TODO: check whether need to set here?
def patch_ray():
    TORCH_NCCL_DTYPE_MAP[torch.bfloat16] = nccl.NCCL_BFLOAT16

patch_ray()


def init_ray(runtime_env_args):
    runtime_env = {"env_vars": {}}
    python_path = os.environ.get("PYTHONPATH", "")
    if python_path:
        runtime_env["env_vars"]["PYTHONPATH"] = python_path

    def _set_runtime_env(runtime_env_args, attribute, runtime_env):
        if getattr(runtime_env_args, attribute):
            runtime_env[attribute] = getattr(runtime_env_args, attribute)

    for key in ['pip', 'working_dir', 'py_modules', 'excludes']:
        _set_runtime_env(runtime_env_args, key, runtime_env)

    # namespace is needed to get NamedActor
    ray.init(runtime_env=runtime_env, namespace="RLHF")


def init(args=None):
    """
    Initialize RLHF env, including
    1. init_process_group for distributed
    2. ...
    """
    if args is None:
        args = parse_args()
    set_global_variables(args)
    if dlc_utils.in_dlc_env():
        dlc_utils.start_ray_cluster()
    init_ray(args.env_args)
    set_initialized()
    if dlc_utils.in_dlc_env():
        dlc_utils.start_exit_listener()
        if dlc_utils.get_rank() > 0:
            # other workers exit after head exit
            sys.exit(0)
    logger.info(f"init rlhf done, rlhf version {VERSION}")
