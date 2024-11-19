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
"""Hooks of vllm-0.6.3 convert device_name to string."""


import inspect
# pylint: disable=unused-import,unused-argument
from vllm.platforms import cuda


source = inspect.getsource(cuda.CudaPlatform.get_device_name)
if 'physical_device_id = device_id_to_physical_device_id(device_id)' in source:
    from vllm.platforms.cuda import device_id_to_physical_device_id, get_physical_device_name

    @classmethod
    def _get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return str(get_physical_device_name(physical_device_id))

    cuda.CudaPlatform.get_device_name = _get_device_name
