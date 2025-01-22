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
"""Hooks of vllm-0.6.3 worker_base to update execute_method."""

# pylint: disable=unused-import,wildcard-import
from vllm.worker import worker_base
from vllm.worker.worker_base import logger


def execute_method(self, method, *args, **kwargs):
    try:
        if self.worker is None:
            target = self
        else:
            if hasattr(self.worker, method):
                target = self.worker
            else:
                target = self
        #print(f"debug target: {target} method: {method}")
        executor = getattr(target, method)
        return executor(*args, **kwargs)
    except Exception as e:
        # if the driver worker also execute methods,
        # exceptions in the rest worker may cause deadlock in rpc like ray
        # see https://github.com/vllm-project/vllm/issues/3455
        # print the error and inform the user to solve the error
        msg = (f"Error executing method {method}. "
                "This might cause deadlock in distributed execution.")
        logger.exception(msg)
        raise e

worker_base.WorkerWrapperBase.execute_method = execute_method
