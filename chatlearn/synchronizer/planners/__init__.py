# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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
"""The collections of planners"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chatlearn.runtime.dist_actor import DistModel

def get_planner_cls(src_model: 'DistModel', dst_model: 'DistModel'):
    # pylint: disable=unused-argument, import-outside-toplevel
    from .planner import MegatronVLLMSyncPlanner
    return MegatronVLLMSyncPlanner
