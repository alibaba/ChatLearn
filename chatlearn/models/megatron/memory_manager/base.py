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
"""Base class for memory managers, and common utilities."""
import gc

import torch

from chatlearn.utils.timer import Timers, _Timer


class BaseMemoryManager:
    """
    Base class of memory managers for Megatron modules which provides utilities to free memory when unused.
    """

    def __init__(self, model, model_name, timers):
        self._model = model
        self._model_name = model_name
        self._timers = timers

    def _wrap_method(self, func, timers: Timers):
        def inner(*args, **kwargs):
            torch.cuda.synchronize()
            timer: _Timer = timers(f'{self._model_name}_free_memory')
            if not timer.started_:
                timer.start()
            func(*args, **kwargs)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            timer.stop()

        return inner
