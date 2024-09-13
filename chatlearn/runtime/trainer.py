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
"""Trainer"""

import math
import ray

from chatlearn.utils import future
from chatlearn.utils.logger import logger
from .executor import Executor
from .utils import encode_data


class Trainer(Executor):
    """Trainer"""

    def __init__(self, model_flow):
        """
        Trainer

        Args
        ----
        models : List[BaseModule]
            a list of modules
        """
        super().__init__(model_flow)
        for model, func_name in self.model_to_call_func.items():
            model.trainable_funcs.append(func_name)
        self.iteration = 0
        self._data_parallel_size = None

    def set_data_loader(self, data_loader):
        self._data_loader = data_loader

    def next_batch(self):
        batches = []
        for _ in range(self.num_micro_batch_per_dp):
            data = self._data_loader.next.remote()
            if future.get(self._data_loader.has_next.remote()):
                batches.append(data)
        if not batches:
            return
        else:
            if len(batches) < self.num_micro_batch_per_dp:
                batches += batches[:self.num_micro_batch_per_dp - len(batches)]
            return batches

    @property
    def num_iteration(self):
        # Given that we have incorporated support for relay buffer and dynamic reward outputs,
        # the number of training data batches per episode may differ, hence we dynamically determine the total number of batches per episode.
        _sample_per_episode = ray.get(self._data_loader.total_samples.remote())
        return math.ceil(_sample_per_episode / self.args.train_global_batch_size)

    @property
    def data_parallel_size(self):
        if self._data_parallel_size is None:
            self._data_parallel_size = self.first_model.replicas[0].data_parallel_size
            for model in self.models[1:]:
                assert model.replicas[0].data_parallel_size == self._data_parallel_size, \
                "Currently, all training models are assumed to have the same data_parallel_size"
        return self._data_parallel_size

    def train(self, episode):
        self.num_micro_batch_per_dp = self.args.train_global_batch_size // self.args.train_micro_batch_size // self.data_parallel_size
        _num_training_iteration = self.num_iteration
        self._batch_per_episode = _num_training_iteration
        for epoch in range(self.args.num_training_epoch):
            if epoch > 0:
                ret = self._data_loader.shuffle.remote()
                future.wait(ret)
            data_queues, out_queue = self.setup_queues()
            for mb in range(_num_training_iteration * self.data_parallel_size):
                batch = encode_data(mb, self.next_batch())
                for data_queue in data_queues:
                    data_queue.put(batch)
            self.compute_loop(out_queue, _num_training_iteration)
            self.iteration = self.iteration + _num_training_iteration
            logger.info(f"train episode: {episode+1}, epoch {epoch} num_step {_num_training_iteration} done")
