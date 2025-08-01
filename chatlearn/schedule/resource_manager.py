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
"""resource manager"""

import os
import time
from typing import Tuple

import ray
import ray.experimental.state.api
from ray.util.placement_group import placement_group

from chatlearn.utils.logger import logger
from chatlearn.models.base_module import BaseModule


class ResourceManager:
    """
    Manage hardware resources for each task.
    Used for managing ray placement_group
    """

    def __init__(self, models: Tuple[BaseModule]):
        self.models = models
        self.name2models = {model.name: model for model in self.models}
        self.model_to_placegroup = {}
        self.placement_groups = []
        resource = ray.nodes()[0]['Resources']
        self.gpu_per_node = int(resource['GPU'])
        self.cpu_per_node = int(resource['CPU'])
        self.nnode = int(os.environ['WORLD_SIZE'])
        self.total_gpu = self.nnode * self.gpu_per_node

    def get_placement_group_state(self, pg):
        try:
            state = ray.experimental.state.api.get_placement_group(pg.id.hex())["state"]
            return state
        except Exception as e:
            logger.warning(f"fail to get placement_group state {e}")

    def create_placement_group(self, num_gpus, num_cpus=None, strategy="PACK"):
        """
        create resource placement group given model device args
        """
        if num_gpus > 0:
            # a resource bundles should in on node
            # each bundle represents a group of reserved resource
            if num_gpus <= self.gpu_per_node:
                cpu_count = int(self.cpu_per_node * num_gpus / self.gpu_per_node)
                bundles = [{"GPU": num_gpus, "CPU": cpu_count}]
            else:
                assert num_gpus % self.gpu_per_node == 0
                num_nodes = num_gpus // self.gpu_per_node
                bundles = [{"GPU": self.gpu_per_node, "CPU": self.cpu_per_node} for _ in range(num_nodes)]
        else:
            assert num_cpus is not None and isinstance(num_cpus, list), "num_cpus should be type of list"
            bundles = [{"GPU": 0, "CPU": num_cpu} for num_cpu in num_cpus]
        pg = placement_group(bundles, strategy=strategy)
        warn_once = True
        while self.get_placement_group_state(pg) == "PENDING":
            if warn_once:
                logger.info(ray.experimental.state.api.list_nodes())
                logger.info(f"waiting for placement group to be created for {num_gpus}GPUs {pg.bundle_specs}")
                warn_once = False
            time.sleep(1)
        self.placement_groups.append(pg)
        return pg

    def remove_placement_groups(self):
        for pg in self.placement_groups:
            ray.util.remove_placement_group(pg)
            while self.get_placement_group_state(pg) != "REMOVED":
                time.sleep(0.5)
        logger.info("Remove placement groups done")
