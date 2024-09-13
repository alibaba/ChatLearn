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
"""shared storage."""

import ray

from chatlearn.utils import future
from chatlearn.utils.logger import logger


@ray.remote
class Storage:
    """Shared storage"""

    def __init__(self):
        self._storage = {}

    def put(self, key, data):
        """
        put data with key
        """
        ref = ray.put(data)
        self._storage[key] = ref

    def get(self, key):
        """
        get data by key
        """
        ref = self._storage.get(key)
        if ref is None:
            logger.warning("%s is not found in storage", key)
            return None
        return future.get(ref)
