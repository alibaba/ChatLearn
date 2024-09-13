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
"""Evaluator"""

from collections import defaultdict
import math

from chatlearn.runtime.environment import Environment
from chatlearn.utils import future
from chatlearn.utils.logger import logger
from chatlearn.data.ranking import batch_generation_ranking

# pylint: disable=not-callable
class Evaluator(Environment):
    """
    Evaluator.

    Args
    ----
    models : [BaseModule]
        models to evaluate
    args : RuntimeConfig
        default to None
    """

    def __init__(self, model_flow):
        super().__init__(model_flow)
        self._post_process_func = None
        self.is_eval = True

    @property
    def sample_per_episode(self):
        return len(self._dataset)

    @property
    def batch_per_episode(self):
        if self._batch_per_episode is not None:
            return self._batch_per_episode
        if self.first_model.use_vllm_backend:
            # For the vLLM model, the number of samples processed at one time is sample_per_episode / num_replicas.
            # If sample_per_episode < num_replicas, then some models will not receive any data to process,
            # in which case batch_per_episode = sample_per_episode (each model processes one sample).
            # If sample_per_episode is greater than num_replicas, then batch_per_episode = num_replicas.
            if self.sample_per_episode >= len(self.models[0].replicas):
                self._batch_per_episode = len(self.models[0].replicas)
            else:
                self._batch_per_episode = self.sample_per_episode
        else:
            self._batch_per_episode = math.ceil(len(self._dataset) / self.batch_size)
        return self._batch_per_episode

    def setup_dataset(self):
        assert len(self._dataset) > 0, "dataset is not set"
        if self.models[0].module_args.batch_generation.ranking:
            logger.info("calling batch_generation_ranking")
            self._dataset = batch_generation_ranking(self._dataset, 1, len(self._dataset))
        refs = []
        for idx, model_replica in enumerate(self.models[0].replicas):
            if self.first_model.use_vllm_backend:
                remainder = self.sample_per_episode % self.models[0].num_replica
                batch_size_plus = 1 if idx < remainder else 0
                batch_size = self.batch_size + batch_size_plus
            else:
                batch_size = self.batch_size
            if batch_size > 0:
                ref = model_replica.master._build_dataloader.remote(
                    self._dataset, batch_size, dynamic_batch_size_flag=self.first_model.use_vllm_backend, is_eval=True)
                refs.append(ref)
        future.get(refs)

    def get_all_merged_data_list(self, queues, encode=True):
        queue0 = queues[0]
        merged_data_list = []
        while queue0.qsize() > 0:
            res = self.get_merged_data(queues, encode)
            merged_data_list.append(res)
        return merged_data_list

    def set_post_process_func(self, post_process_func):
        """
        Set post process function for model evaluation results.

        Args
        ----
        post_process_func

            This function accept two arguments.
            1. results: a list of evaluation results
            2. eval_info: a dict meta that contains "train_iteration" and "episode_iteration"
        """
        self._post_process_func = post_process_func
        return self

    def eval(self, cur_iter=None, train_iteration=None):
        """
        Evaluating.

        Args
        ----
        cur_iter : int
            current iteration.
        train_iteration: int
            current training iteration.
        """
        refs = []
        for model in self.models[0].replicas:
            refs.append(model.master.reset_eval_data_iter.remote())
        future.get(refs)
        out_queue = self.execute(is_eval=True)
        queue_size = out_queue.qsize()
        result_refs = [out_queue.get() for _ in range(queue_size)]
        element_size = len(result_refs[0])
        data_list = future.wait(result_refs, desc="evaluator", return_output=True)
        results = [data_list[i:i + element_size] for i in range(0, len(data_list), element_size)]
        all_results = defaultdict(list)
        for batches in results:
            for i, batch in enumerate(batches):
                model_name = self.model_flow.return_model_nodes[i].name
                all_results[model_name].append(batch)

        if self._post_process_func is not None:
            eval_info = {}
            if cur_iter is not None:
                eval_info["episode_iteration"] = cur_iter
            if train_iteration is not None:
                eval_info["train_iteration"] = train_iteration
            self._post_process_func(all_results, eval_info)
        return all_results
# pylint: disable=not-callable
