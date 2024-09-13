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
"""Environment"""

import math
from itertools import cycle

from chatlearn.data.ranking import batch_generation_ranking
from chatlearn.utils import future
from chatlearn.utils.logger import logger
from .executor import Executor
from .utils import encode_data

# pylint: disable=not-callable
class Environment(Executor):
    """BaseEnv"""

    def __init__(self, model_flow):
        """
        Environment

        Args
        ----
        models : List[BaseModule]
            a list of modules
        """
        super().__init__(model_flow)
        self._batch_size = None
        self._batch_per_episode = None
        self._dataset = None
        self.data_iter = None
        self._padding_config = {}
        self.merged_buffer = {}
        self.model2iter = {}

    def set_dataset(self, dataset):
        self._dataset = dataset
        return self

    def setup_dataset(self):
        self.data_producer = self.models[0]
        assert self.sample_per_episode % len(self.data_producer.replicas) == 0, \
            "replica number of data producer model must be divisible by sample_per_episode"
        logger.info("start set dataset for data_producer")
        refs = []
        if self.models[0].module_args.batch_generation.ranking:
            episode_per_epoch = math.ceil(len(self._dataset) / self.sample_per_episode)
            self._dataset = batch_generation_ranking(self._dataset, episode_per_epoch, self.sample_per_episode)
        for policy_replica in self.data_producer.replicas:
            ref = policy_replica.master._build_dataloader.remote(self._dataset,
                                                                 self.batch_size)
            refs.append(ref)
        future.get(refs)
        logger.info("set dataset for data_producer done")

    def setup(self):
        super().setup()
        self.setup_dataset()

        for model_node in self.model_flow.model_nodes:
            model = model_node.model.replicas[0]
            config = future.get(model.master.padding_config.remote())
            self._padding_config.update(config)

    @property
    def sample_per_episode(self):
        return self.args.sample_per_episode

    @property
    def batch_size(self):
        if self._batch_size is not None:
            return self._batch_size
        if self.first_model.use_vllm_backend:
            num_replica = len(self.models[0].replicas)
            self._batch_size = self.sample_per_episode // num_replica
        else:
            self._batch_size = self.models[0].module_args.generation_batch_size

        return self._batch_size

    @property
    def batch_per_episode(self):
        if self._batch_per_episode is not None:
            return self._batch_per_episode
        num_replica = len(self.models[0].replicas)
        num_batch = self.sample_per_episode // (num_replica * self.batch_size) * num_replica
        remainder = self.sample_per_episode % (num_replica * self.batch_size)
        if remainder >= num_replica:
            self._batch_per_episode = num_batch + num_replica
        else:
            self._batch_per_episode = num_batch + remainder
        return self._batch_per_episode

    @property
    def num_iteration(self):
        if self.models[0].module_args.zero_size > 1:
            assert self.batch_per_episode % self.models[0].module_args.zero_size == 0
            return self.batch_per_episode // self.models[0].module_args.zero_size
        else:
            return self.batch_per_episode

    def execute(self, is_eval):
        data_queues, out_queue = self.setup_queues()
        data_producer_iter = cycle(iter(self.models[0].replicas))
        # prepare batches for all model replicas
        for mb in range(self.batch_per_episode):
            current_data_producer = next(data_producer_iter)
            query = current_data_producer.master.next_batch.remote(is_eval=is_eval)
            encoded_data = encode_data(mb, query)
            for data_queue in data_queues:
                data_queue.put(encoded_data)
        self.compute_loop(out_queue, self.num_iteration)
        return out_queue

    def make_experiences(self):
        """
        Generate a collection of experiences for one episode
        """
        return self.execute(is_eval=False)
# pylint: disable=not-callable
