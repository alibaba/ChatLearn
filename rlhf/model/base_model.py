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
"""Base Task."""

class BaseModel:
    """
    abstract model
    """

    def __init__(self, device_count, name, trainable=False):
        self.device_count = device_count
        self.name = name
        self.is_first_rank = None
        self.global_ranks = None
        self.world_size = None
        self.global_ranks = None
        self.global_rank = None
        self.rank = None
        self.group_id = None
        self.trainable = trainable
        # whether this model needs to be executed in current rank
        self.active = False


    def forward(self):
        """
        forward func
        """
        # TODO(sayang): Mock function
        if not self.active:
            return
        if self.is_first_rank:
            print("model {} group {} global rank {} run forward".format(self.name, self.group_id, self.global_rank))


    def backward(self):
        """
        backward func
        """
        # TODO(sayang): Mock function
        if not self.active or not self.trainable:
            return
        if self.is_first_rank:
            print("model {} group {} global rank {} run backward".format(self.name, self.group_id, self.global_rank))
