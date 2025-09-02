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

import ray
from ray.actor import ActorHandle

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
        for model, func_names in self.model_to_call_funcs.items():
            model.trainable_funcs += func_names
        self.iteration = 0
        self._data_parallel_size = None

    def setup(self):
        super().setup()
        for model_node in self.model_flow.model_nodes:
            model_node.trainable = True

    def set_data_loader(self, data_loader: ActorHandle):
        self._data_loader = data_loader

    def next_batch(self):

        return self._data_loader.next.remote()

    # pylint: disable=unused-argument
    def num_iteration(self, model=None) -> int:
        """
        return the number of times the model is updated per episode.
        """
        # Given that we have incorporated support for replay buffer and dynamic reward outputs,
        # the number of training data batches per episode may differ, hence we dynamically determine the total number of batches per episode.
        _num_iteration = ray.get(self._data_loader.num_iteration.remote())
        return _num_iteration

    @property
    def data_parallel_size(self) -> int:
        if self._data_parallel_size is None:
            self._data_parallel_size = self.first_model.replicas[0].data_parallel_size
            for model in self.models[1:]:
                assert model.replicas[0].data_parallel_size == self._data_parallel_size, \
                "Currently, all training models are assumed to have the same data_parallel_size"
        return self._data_parallel_size

    def train(self, episode: int):
        """
        _num_training_iteration(int): The number of times the model is updated per episode
        """
        _num_training_iteration = self.num_iteration()
        self._batch_per_episode = _num_training_iteration
        future.wait(self._data_loader.set_dp_size.remote(self.data_parallel_size))
        for epoch in range(self.args.num_training_epoch):

            # shuffle data
            if epoch > 0:
                future.wait(self._data_loader.shuffle.remote())

            data_queues, out_queue = self.setup_queues()
            batch_list = []
            # Data will merge by dp_rank order after environment.make_experiences()
            # For off-policy, we need to get batch by iterate over dp first, iteration second
            for dp_rank in range(self.data_parallel_size):
                for iter_ in range(_num_training_iteration):
                    batch = encode_data(iter_ * self.data_parallel_size + dp_rank, self.next_batch())
                    batch_list.append(batch)

            # After get all batches, put batch into trainer's input queue by iterate over iteration first, dp second
            for iter_ in range(_num_training_iteration):
                for dp_rank in range(self.data_parallel_size):
                    for data_queue in data_queues:
                        data_queue.put(batch_list[dp_rank * _num_training_iteration + iter_])

            self.compute_loop(out_queue, _num_training_iteration)
            self.iteration = self.iteration + _num_training_iteration
            logger.info(f"train episode: {episode+1}, epoch {epoch} num_step {_num_training_iteration} done")
