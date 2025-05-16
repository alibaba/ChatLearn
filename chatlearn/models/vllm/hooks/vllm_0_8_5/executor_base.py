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
"""Hooks of vllm-0.7.4 executor_base sleep and wake_up for param_sync or generation."""


import os
import time

# pylint: disable=unused-import,wildcard-import,unused-argument,use-dict-literal
from vllm.executor import executor_base
from chatlearn.utils.logger import logger

def sleep(self, level: int = 1):
    if self.is_sleeping:
        logger.info("Executor is already sleeping.")
        return
    is_param_sync = int(os.getenv('CHATLEARN_PARAM_SYNC_STAGE', '0')) > 0
    time_before_sleep = time.perf_counter()
    self.collective_rpc("sleep", kwargs=dict(level=level, is_param_sync=is_param_sync))
    time_after_sleep = time.perf_counter()
    self.is_sleeping = True
    logger.info("It took %.6f seconds to fall asleep.",
                time_after_sleep - time_before_sleep)

executor_base.ExecutorBase.sleep = sleep

def wake_up(self):
    if not self.is_sleeping:
        logger.info("Executor is not sleeping.")
        return
    is_param_sync = int(os.getenv('CHATLEARN_PARAM_SYNC_STAGE', '0')) > 0
    time_before_wakeup = time.perf_counter()
    self.collective_rpc("wake_up", kwargs=dict(is_param_sync=is_param_sync))
    time_after_wakeup = time.perf_counter()
    self.is_sleeping = False
    logger.info("It took %.6f seconds to wake up.",
                time_after_wakeup - time_before_wakeup)

executor_base.ExecutorBase.wake_up = wake_up
