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

from itertools import cycle
from typing import Optional

from chatlearn.models.vllm_module import VLLMModule
from chatlearn.runtime.dist_actor import DistModel
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
        """
        super().__init__(model_flow)
        self._batch_size = None
        self._batch_per_episode = None
        self._all_datasets = None
        self.data_iter = None

    def set_dataset(self, dataset):
        """Set dataset for the environment.

        Args:
            dataset (list): a list of prompts strs

        Returns:
            Environment instance: return environment
        """
        assert isinstance(dataset, list), (
            f"expect the dataset to be a list of prompts, got {type(dataset)}"
        )
        assert not isinstance(dataset[0], list), (
            "expect only one dataset to be set, if you want to use more "
            "than one dataset, please try `set_multiple_datasets`"
        )
        self._all_datasets = [dataset]
        return self

    def set_multiple_datasets(self, all_datasets):
        """Set multiple datasets for the environment.

        Args:
            dataset (list): a list of prompts strs

        Returns:
            Environment instance: return environment
        """
        # sanity check
        assert len(all_datasets) >= 1, (
            f"expect at least one dataset, got {len(all_datasets)} datasets."
        )
        assert isinstance(all_datasets, list), (
            f"expect datasets to be a list, got {type(all_datasets)}"
        )
        for dataset in all_datasets:
            assert isinstance(dataset, list), (
                f"expect each dataset to be a list of prompts, got {type(dataset)}"
            )

        self._all_datasets = all_datasets
        return self

    def setup_dataset(self):
        self.data_producer = self.models[0]
        assert self.sample_per_episode % len(self.data_producer.replicas) == 0, \
            "replica number of data producer model must be divisible by sample_per_episode"
        logger.info("start set dataset for data_producer")
        refs = []
        for policy_replica in self.data_producer.replicas:
            ref = policy_replica.master._build_dataloader.remote(self._all_datasets,
                                                                 self.sample_per_episode)
            refs.append(ref)
        future.get(refs)
        logger.info("set dataset for data_producer done")

    def setup(self):
        super().setup()
        self.setup_dataset()

        for model_node in self.model_flow.model_nodes:
            model = model_node.model.replicas[0]
            if model.model.name == "policy":
                logger.info(f"setup engine for rollout {model.model}")
                refs = []
                for replica in model_node.model.replicas:
                    if isinstance(model.model, VLLMModule):
                        refs.append(replica.setup_engine(replica.all_actors))
                    else:
                        refs.append(replica.setup_engine())
                future.wait(refs, return_output=True)

    @property
    def sample_per_episode(self):
        return self.args.sample_per_episode

    def batch_size(self, model=None):
        if model is None:
            model = self.models[0]

        if model.use_vllm_backend:
            num_replica = len(model.replicas)
            batch_size = self.sample_per_episode // num_replica
        else:
            batch_size = model.module_args.generation_batch_size

        return batch_size

    def global_dp_size(self, model: Optional[DistModel]=None) -> int:
        """
        !!! it seems just return num_replica * dp_size
        the number is used for split data in Environment.execute
        """
        if model is None:
            model = self.models[0]

        num_replica = len(model.replicas)
        dp_size = len(model.replicas[0].dp_rank_to_actors)
        assert self.sample_per_episode >= num_replica, "sample_per_episode need larger than num replica"
        assert dp_size >= 1, "dp size need larger or equal to 1"

        return num_replica * dp_size

    def num_iteration(self, model: Optional[DistModel]=None):
        """Calculate the number of iterations for a model in the environment.
        !!! It seems equal to num_replica
        """
        if model is None:
            model = self.models[0]

        # !!! it seems just return num_replica
        return self.global_dp_size(model) // len(model.replicas[0].dp_rank_to_actors)

    def execute(self, is_eval):
        data_queues, out_queue = self.setup_queues()
        data_producer_iter = cycle(iter(self.models[0].replicas))
        # prepare batches for all model replicas
        for mb in range(self.global_dp_size(self.models[0])):
            current_data_producer = next(data_producer_iter)
            # master is vllm.engine or self.all_actors[0]
            query = current_data_producer.master.next_batch.remote(is_eval=is_eval)
            encoded_data = encode_data(mb, query)
            for data_queue in data_queues:
                data_queue.put(encoded_data)
        self.compute_loop(out_queue)
        return out_queue

    def make_experiences(self):
        """
        Generate a collection of experiences for one episode
        """
        return self.execute(is_eval=False)
        