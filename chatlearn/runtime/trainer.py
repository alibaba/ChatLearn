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
"""Trainer"""

import math
import ray

from chatlearn.utils import future
from chatlearn.utils.logger import logger
from .executor import Executor


class Trainer(Executor):
    """Trainer"""

    def __init__(self, models):
        """
        Trainer

        Args
        ----
        models : List[RLHFModule]
            a list of modules
        """
        super().__init__(models)
        self.models = models
        for model in models:
            model.register_func("train_step")
        self.num_micro_batch = self.args.train_global_batch_size // self.args.train_micro_batch_size
        self.iteration = 0

    def set_data_loader(self, data_loader):
        self._data_loader = data_loader

    def next_batch(self):
        batches = []
        for _ in range(self.num_micro_batch):
            data = self._data_loader.next.remote()
            if future.get(self._data_loader.has_next.remote()):
                batches.append(data)
        if not batches:
            return
        else:
            if len(batches) < self.num_micro_batch:
                batches += batches[:self.num_micro_batch - len(batches)]
            return batches

    def num_training_iteration(self):
        # Given that we have incorporated support for relay buffer and dynamic reward outputs,
        # the number of training data batches per episode may differ, hence we dynamically determine the total number of batches per episode.
        _sample_per_episode = ray.get(self._data_loader.total_samples.remote())
        return math.ceil(_sample_per_episode / self.args.train_global_batch_size)

    def train(self, episode):
        _num_training_iteration = self.num_training_iteration()
        self._batch_per_episode = _num_training_iteration
        for epoch in range(self.args.num_training_epoch):
            if epoch > 0:
                ret = self._data_loader.shuffle.remote()
                future.wait(ret)
            data_queues, out_queue = self.setup_queues()
            for mb in range(_num_training_iteration):
                batch = self.encode_data(mb, self.next_batch())
                for data_queue in data_queues:
                    data_queue.put(batch)
            self.compute_loop(data_queues, out_queue, _num_training_iteration)
            self.iteration = self.iteration + _num_training_iteration
            logger.info(f"train episode: {episode}, epoch {epoch} num_step {_num_training_iteration} done")
