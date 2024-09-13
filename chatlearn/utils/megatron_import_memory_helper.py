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
"""
Version compatibility utilities for Megatron memory management of gradients and parameter weights.
Base on how Megatron uses buffers to manage memory, we support 3 different versions.
"""
from enum import Enum, auto
from typing import List

__all__ = ['MegatronVersion', 'get_megatron_version', 'check_megatron_versions']


class MegatronVersion(Enum):
    """
    There are currently three different Megatron versions supported.
    """

    V1 = auto()  # use `MemoryBuffer` to manage gradients
    V2 = auto()  # use `GradBuffer` to manage gradients
    V3 = auto()  # use `ParamAndGradBuffer` to manage parameter weights and gradients


def get_megatron_version():
    try:
        # pylint: disable-next=import-outside-toplevel, unused-import
        from megatron.core.distributed import ParamAndGradBuffer

        return MegatronVersion.V3
    except ImportError:
        ...
    try:
        # pylint: disable-next=import-outside-toplevel, unused-import
        from megatron.core.distributed import GradBuffer

        return MegatronVersion.V2
    except ImportError:
        ...
    return MegatronVersion.V1


def check_megatron_versions(targets: List[MegatronVersion]):
    version = get_megatron_version()
    assert version in targets, f'Different Megatron version {version} from targets: {targets}.'


_version = get_megatron_version()

# pylint: disable=unused-import

if _version == MegatronVersion.V3:
    from megatron.core.distributed.param_and_grad_buffer import BufferType

    __all__.append('BufferType')

# pylint: enable=unused-import
