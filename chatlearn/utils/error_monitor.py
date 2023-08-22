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
"""Error monitor"""

import time

import ray
import ray.util.collective as col

from chatlearn.utils import future
from chatlearn.utils.logger import logger


@ray.remote
class ErrorMonitor:
    """Error Monitor"""

    def __init__(self, error_signal, remote_models, group_names):
        self.error_signal = error_signal
        self.remote_models = remote_models
        self.collective_groups = group_names

    def monitor(self):
        while True:
            catch_err = future.get(self.error_signal.is_set.remote())
            if catch_err:
                error_msg = future.get(self.error_signal.error_msg.remote())
                break
            time.sleep(2)
        logger.exception(f"Error found {error_msg}")
        for group_name in self.collective_groups:
            col.destroy_collective_group(group_name)
        for model in self.remote_models:
            model.terminate()
        try:
            exit_actor = ray.get_actor("ExitActor")
            ray.kill(exit_actor)
        except Exception:
            pass
        ray.shutdown()


@ray.remote(num_cpus=0)
class ErrorSignalActor:
    """ErrorSignalActor"""
    def __init__(self):
        self.error_state = False
        self.err_msg = None

    def set(self, err_msg=None):
        self.error_state = True
        if err_msg is not None:
            self.err_msg = err_msg

    def is_set(self):
        return self.error_state

    def error_msg(self):
        return self.err_msg