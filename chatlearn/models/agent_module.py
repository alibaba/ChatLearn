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
"""agent module"""
from typing import Dict, List

from chatlearn import BaseModule

class AgentModule(BaseModule):
    """Agent Module"""

    def __init__(self, name: str, args=None, replica_id: int=0):
        """ChatLearn main agent entrypoint
        """
        super().__init__(name, args=args, replica_id=replica_id)
        assert self.total_gpu == 0, "AgentModule does not require GPU"
        self._num_gpu_per_replica = 0
        self._num_replica = self.module_args.num_cpu // self.module_args.cpu_per_process

    def setup(self):
        self.stats = {}
        self._metric_prefix = "agent"

    def _forward_step(self, data: List[Dict]):
        return data

    def forward_step(self, data: List[Dict], iteration=0, **kwargs):
        return data

    def eval_forward(self, data: List[Dict], **kwargs):
        return data
    