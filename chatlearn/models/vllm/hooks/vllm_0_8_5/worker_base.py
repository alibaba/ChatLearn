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
"""Hooks of vllm-0.6.6 worker_base to update execute_method."""

# pylint: disable=unused-import,wildcard-import
from typing import Union
from vllm.worker import worker_base
from vllm.worker.worker_base import logger
from vllm.utils import run_method


del worker_base.WorkerWrapperBase.__getattr__
def execute_method(self, method: Union[str, bytes], *args, **kwargs):
    try:
        if self.worker is None:
            target = self
        else:
            if hasattr(self, method):
                target = self
            else:
                target = self.worker
        # method resolution order:
        # if a method is defined in this class, it will be called directly.
        # otherwise, since we define `__getattr__` and redirect attribute
        # query to `self.worker`, the method will be called on the worker.
        return run_method(target, method, args, kwargs)
    except Exception as e:
        # if the driver worker also execute methods,
        # exceptions in the rest worker may cause deadlock in rpc like ray
        # see https://github.com/vllm-project/vllm/issues/3455
        # print the error and inform the user to solve the error
        msg = (f"Error executing method {method!r}. "
                "This might cause deadlock in distributed execution.")
        logger.exception(msg)
        raise e

worker_base.WorkerWrapperBase.execute_method = execute_method
