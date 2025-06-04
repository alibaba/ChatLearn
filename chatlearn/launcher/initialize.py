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
from ray.dag.compiled_dag_node import ExecutableTask
from ray.dag.dag_node_operation import _DAGNodeOperationType

from chatlearn.launcher import dlc_utils
from chatlearn.utils.arguments import parse_args
from chatlearn.utils.global_vars import set_global_variables
from chatlearn.utils.logger import logger
from chatlearn.utils.version import VERSION



def patch_ray():
    TORCH_NCCL_DTYPE_MAP[torch.bfloat16] = nccl.NCCL_BFLOAT16
    if ray.__version__ == "2.46.0":
        # vllm 0.8.5.post1 will install ray 2.46.0 and need patch function
        def exec_operation(
            self,
            class_handle,
            op_type: _DAGNodeOperationType,
            overlap_gpu_communication: bool = False,
        ) -> bool:
            """
            An ExecutableTask corresponds to a DAGNode. It consists of three
            operations: READ, COMPUTE, and WRITE, which should be executed in
            order to ensure that each operation can read the correct intermediate
            result.
            Args:
                class_handle: The handle of the class to which the actor belongs.
                op_type: The type of the operation. Possible types are READ,
                    COMPUTE, and WRITE.
                overlap_gpu_communication: Whether to overlap GPU communication with
                    computation during DAG execution to improve performance.
            Returns:
                True if the next operation should not be executed; otherwise, False.
            """
            if op_type == _DAGNodeOperationType.READ:
                # with _device_context_manager():
                with self._recv_stream:
                    return self._read(overlap_gpu_communication)
            elif op_type == _DAGNodeOperationType.COMPUTE:
                return self._compute(overlap_gpu_communication, class_handle)
            elif op_type == _DAGNodeOperationType.WRITE:
                # with _device_context_manager():
                with self._send_stream:
                    return self._write()
        ExecutableTask.exec_operation = exec_operation


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
    if dlc_utils.in_dlc_env():
        listener = dlc_utils.StartExitListener()
        listener.start_exit_listener()
        if dlc_utils.get_rank() > 0:
            logger.info(f"RANK: {dlc_utils.get_rank()}: task finish, exit ...")
            # other workers exit after head exit
            sys.exit(0)
    logger.info(f"init chatlearn done, version {VERSION}")
