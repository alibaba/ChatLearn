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
"""port manager"""

from multiprocessing import Lock
import ray

@ray.remote
class PortManager:
    """port manager"""

    def __init__(self, port_list):
        self._port_list = port_list
        self._address_to_port_index = {}
        self._lock = Lock()

    def get_free_port(self, address):
        self._lock.acquire()
        free_port = None
        try:
            port_index = self._address_to_port_index.get(address, 0)
            assert port_index < len(self._port_list)
            self._address_to_port_index[address] = port_index + 1
            free_port = self._port_list[port_index]
        finally:
            self._lock.release()
        return free_port
