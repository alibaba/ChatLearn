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
"""Initialize"""

import os
import sys

import ray
import torch
from cupy.cuda import nccl
from ray.util.collective.collective_group.nccl_util import TORCH_NCCL_DTYPE_MAP

from chatlearn.launcher import dlc_utils
from chatlearn.utils.arguments import parse_args
from chatlearn.utils.global_vars import set_global_variables
from chatlearn.utils.global_vars import set_initialized
from chatlearn.utils.logger import logger
from chatlearn.utils.version import VERSION


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
    ray.init(runtime_env=runtime_env, namespace="CHATLEARN", _node_ip_address=dlc_utils.get_addr(), log_to_driver=False)


def init(args=None):
    """
    Initialize ChatLearn env, including
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
        listener = dlc_utils.StartExitListener()
        listener.start_exit_listener()
        if dlc_utils.get_rank() > 0:
            logger.info(f"RANK: {dlc_utils.get_rank()}: task finish, exit ...")
            # other workers exit after head exit
            sys.exit(0)
    logger.info(f"init chatlearn done, version {VERSION}")
