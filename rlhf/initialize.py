import os
import ray
import sys
from rlhf import dlc_utils
from rlhf.arguments import parse_args
from rlhf.global_vars import set_global_variables
from rlhf.global_vars import set_initialized


def init_ray(runtime_env_args):
    runtime_env = {"env_vars": {}}
    python_path = os.environ.get("PYTHONPATH", "")
    if python_path:
        runtime_env["env_vars"]["PYTHONPATH"] = python_path

    def _set_runtime_env(runtime_env_args, attribute, runtime_env):
        if getattr(runtime_env_args, attribute):
            runtime_env[attribute] = getattr(runtime_env_args, attribute)

    _set_runtime_env(runtime_env_args, 'pip', runtime_env)
    _set_runtime_env(runtime_env_args, 'working_dir', runtime_env)
    _set_runtime_env(runtime_env_args, 'py_modules', runtime_env)

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
    if args.env_args.platform == "DLC":
        dlc_utils.start_ray_cluster()
    init_ray(args.env_args)
    set_initialized()
    if args.env_args.platform == "DLC":
        dlc_utils.start_exit_listener()
        if dlc_utils.get_rank() > 0:
            # other workers exit after head exit
            sys.exit(0)
