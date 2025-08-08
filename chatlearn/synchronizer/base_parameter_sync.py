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
"""Abstract class for parameter synchronization."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chatlearn.runtime.dist_actor import DistModel


class BaseParameterSyncGroup(ABC):
    """The ABC for Parameter Synchronization"""
    def __init__(
        self,
        src_model: 'DistModel',
        dst_model: 'DistModel',
        frequency: int,
    ):
        """Manage Parameter Synchronization between source and destination models.

        Args:
            src_model (DistModel): The source distmodel, only MegatronModel is supported.
            dst_model (DistModel): The destination distmodel, only vLLM backend is supported.
            frequency (int): The synchronization frequency of this group. Should be a positve
            integer.
        """
        self.src_model, self.dst_model = src_model, dst_model
        self.frequency = frequency

    @abstractmethod
    def sync(self, dryrun: bool=False):
        """Perform parameter synchronization on this group. If `dryrun` is True,
        some initialization will be excuted and no actual synchronization 
        will be done.

        Args:
            dryrun (bool, optional): Whether to run in dryrun mode. 
            Defaults to False.
        """
