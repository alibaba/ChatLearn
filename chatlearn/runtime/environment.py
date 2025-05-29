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
from chatlearn.models.vllm_module import VLLMModule
from chatlearn.utils import future
from chatlearn.utils.logger import logger
from chatlearn.utils.utils import execute_in_parallel
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
        self._all_datasets = None
        self.data_iter = None
        self._padding_config = {}

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
        if self.models[0].module_args.batch_generation.ranking:
            for i, dataset in enumerate(self._all_datasets):
                episode_per_epoch = math.ceil(len(dataset) / self.sample_per_episode)
                self._all_datasets[i] = batch_generation_ranking(
                    dataset, episode_per_epoch, self.sample_per_episode
                )

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
            config = future.get(model.master.padding_config.remote())
            self._padding_config.update(config)

            if isinstance(model.model, VLLMModule):
                logger.info(
                    f"setup vllm engine for model {model.model}")
                refs = []
                for replica in model_node.model.replicas:
                    refs.append(replica.vllm_engine.setup_vllm.remote(
                        replica.all_actors))
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

    def batch_per_episode(self, model=None):
        if model is None:
            model = self.models[0]

        num_replica = len(model.replicas)
        if self.sample_per_episode >= num_replica:
            _batch_per_episode = num_replica
        else:
            _batch_per_episode = self.sample_per_episode
        dp_size = len(model.replicas[0].dp_rank_to_actors)
        if dp_size > 1:
            _batch_per_episode *= dp_size


        return _batch_per_episode

    def num_iteration(self, model=None):
        """Calculate the number of iterations for a model in the environment.

        Args:
            model: an model in environment. if None, use the first model. default: None.

        Returns:
            The number of iterations for the model in the environment
        """
        if model is None:
            model = self.models[0]

        _batch_per_episode = self.batch_per_episode(model)
        dp_size = len(model.replicas[0].dp_rank_to_actors)
        if model.module_args.zero_size > 1:
            assert _batch_per_episode % model.module_args.zero_size == 0
            return _batch_per_episode // model.module_args.zero_size
        elif dp_size > 1: # for trainable model or ep model
            if _batch_per_episode < dp_size:
                raise NotImplementedError(
                    "Currently ChaLearn requires batch_per_episode >= len(dp_rank_to_actors), "
                    f"got {_batch_per_episode} and {dp_size}. "
                    f"Please allocate more replicas to inference model {model.name} to walk-around the issue."
                )
            assert _batch_per_episode % dp_size == 0, (
                "Inner loop in Executor.generate_step_one_model_internal() depends on dp_size of each replica."
            )
            return _batch_per_episode // dp_size
        else:
            return _batch_per_episode

    def execute(self, is_eval):
        data_queues, out_queue = self.setup_queues()
        data_producer_iter = cycle(iter(self.models[0].replicas))
        # prepare batches for all model replicas
        for mb in range(self.batch_per_episode(self.models[0])):
            current_data_producer = next(data_producer_iter)
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

class MCTSEnv(Environment):
    """MCTS Env"""

    def __init__(self, model_flow, mcts):
        super().__init__(model_flow)
        self.max_iteration_per_sample = self.args.max_iteration_per_sample
        self.mcts = mcts
        assert self.args.sample_per_episode == mcts.module_args.num_cpu

    def mcts_loop(self, max_iteration, encoded_data, data_queues, mb, replica_data_list, mcts):
        future.wait(mcts.init_tree())
        for i in range(max_iteration):
            for data_queue in data_queues:
                data_queue.put(encoded_data)
            for replica, model_node in replica_data_list:
                in_queue = model_node.get_input_queues()
                func_name = model_node.func_name
                # TODO: we will consider colocation/offload later
                to_empty_cache = False
                to_onload = False
                to_offload = False
                self.generate_step_one_model(model_node, replica, in_queue, model_node.out_queues, i, func_name, to_empty_cache,
                                             is_eval=self.is_eval, to_onload=to_onload, to_offload=to_offload, micro_batch_index=mb)
            should_stop = future.get(mcts.should_stop())
            assert len(should_stop) == 1
            if should_stop[0]:
                break

    def execute(self, is_eval):
        data_queues, out_queue = self.setup_queues()
        data_producer_iter = cycle(iter(self.models[0].replicas))
        args = []
        for mb in range(self.batch_per_episode()):
            current_data_producer = next(data_producer_iter)
            query = current_data_producer.master.next_batch.remote(is_eval=is_eval)
            encoded_data = encode_data(mb, query)
            replica_data_list = []
            model_to_replica = {}
            for model_group in self.model_flow.flow_topology:
                for model_node in model_group:
                    model = model_node.model
                    assert not model.is_colocate, "colocation is currently not supported in MCTSEnv"
                    assert not model.enable_offload, "offload is currently not supported in MCTSEnv"
                    if model in model_to_replica:
                        replica = model_to_replica[model]
                    else:
                        replica = self._next_model(model)
                        model_to_replica[model] = replica
                    replica_data_list.append((replica, model_node))
            mcts = [replica_data[0] for replica_data in replica_data_list if replica_data[0].model is self.mcts]
            assert len(mcts) > 0
            mcts = mcts[0]
            args.append((self.max_iteration_per_sample, encoded_data, data_queues, mb, replica_data_list, mcts))
        if self.args.debug:
            for arg in args:
                self.mcts_loop(*arg)
        else:
            execute_in_parallel(self.mcts_loop, args)
        data = [None] * len(self.model_flow.return_model_nodes)
        for model_node in self.model_flow.model_nodes:
            if model_node in self.model_flow.return_model_nodes:
                # let the results order follow model_node order
                data[self.model_flow.return_model_nodes.index(model_node)] = model_node.out_queues[-1]
        if data:
            self.get_all_merged_data(data, out_queue, encode=False)
        return out_queue

class SPRLEnv(Environment):
    """SPRL(Self-Play Reinforcement Learning) Env"""

    def __init__(self, model_flow, sprl):
        super().__init__(model_flow)
        self.max_iteration_per_sample = self.args.max_iteration_per_sample
        self.sprl = sprl

    def sprl_loop(self, max_iteration, encoded_data, data_queues, mb, replica_data_list, sprl):
        future.wait(sprl.reset())
        for i in range(max_iteration):
            for data_queue in data_queues:
                data_queue.put(encoded_data)
            for replica, model_node in replica_data_list:
                in_queue = model_node.get_input_queues()
                func_name = model_node.func_name
                # TODO: we will consider colocation/offload later
                to_empty_cache = False
                to_onload = False
                to_offload = False
                self.generate_step_one_model(model_node, replica, in_queue, model_node.out_queues, i, func_name, to_empty_cache,
                                             is_eval=self.is_eval, to_onload=to_onload, to_offload=to_offload, micro_batch_index=mb)
            should_stop = future.get(sprl.should_stop())
            assert len(should_stop) == 1
            if should_stop[0]:
                break

    def execute(self, is_eval):
        data_queues, out_queue = self.setup_queues()
        data_producer_iter = cycle(iter(self.models[0].replicas))
        args = []
        for mb in range(self.batch_per_episode()):
            current_data_producer = next(data_producer_iter)
            query = current_data_producer.master.next_batch.remote(is_eval=is_eval)
            encoded_data = encode_data(mb, query)
            replica_data_list = []
            model_to_replica = {}
            for model_group in self.model_flow.flow_topology:
                for model_node in model_group:
                    model = model_node.model
                    assert not model.is_colocate, "colocation is currently not supported in SPRLEnv"
                    assert not model.enable_offload, "offload is currently not supported in SPRLEnv"
                    if model in model_to_replica:
                        replica = model_to_replica[model]
                    else:
                        replica = self._next_model(model)
                        model_to_replica[model] = replica
                    replica_data_list.append((replica, model_node))
            sprl = [replica_data[0] for replica_data in replica_data_list if replica_data[0].model is self.sprl]
            assert len(sprl) > 0
            sprl = sprl[0]
            args.append((self.max_iteration_per_sample, encoded_data, data_queues, mb, replica_data_list, sprl))
        # not support execute_in_parallel now.
        for arg in args:
            self.sprl_loop(*arg)
        data = [None] * len(self.model_flow.return_model_nodes)
        for model_node in self.model_flow.model_nodes:
            if model_node in self.model_flow.return_model_nodes:
                # let the results order follow model_node order
                data[self.model_flow.return_model_nodes.index(model_node)] = model_node.out_queues[-1]
        if data:
            self.get_all_merged_data(data, out_queue, encode=False)
        return out_queue

# pylint: disable=not-callable
