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
import ray

from chatlearn.runtime.environment import Environment
from chatlearn.utils import future
from chatlearn.utils.utils import map_reduce_metrics

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
        self.is_eval = True
        self._metric_prefix = "eval"
        self._metric_list = []

    @property
    def sample_per_episode(self):
        return sum(len(dataset) for dataset in self._all_datasets)

    def setup_dataset(self):
        assert len(self._all_datasets) > 0, "dataset is not set"
        for i, dataset in enumerate(self._all_datasets):
            assert len(dataset) > 0, f"dataset {i} is not set"

        refs = []
        for idx, model_replica in enumerate(self.models[0].replicas):
            if self.first_model.use_vllm_backend:
                remainder = self.sample_per_episode % self.models[0].num_replica
                batch_size_plus = 1 if idx < remainder else 0
                batch_size = self.batch_size() + batch_size_plus
            else:
                batch_size = self.batch_size()
            if batch_size > 0:
                ref = model_replica.master._build_dataloader.remote(
                    self._all_datasets, self.sample_per_episode, is_eval=True)
                refs.append(ref)
        future.get(refs)

    def get_all_merged_data_list(self, queues, encode=True):
        queue0 = queues[0]
        merged_data_list = []
        while queue0.qsize() > 0:
            res = self.get_merged_data(queues, encode)
            merged_data_list.append(res)
        return merged_data_list

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
        if isinstance(result_refs[0][0], ray.ObjectRef):
            data_list = future.wait(result_refs, desc="evaluator", return_output=True)
        else:
            data_list = result_refs[0] # List[Dict]
        results = [data_list[i:i + element_size] for i in range(0, len(data_list), element_size)]
        all_results = defaultdict(list)
        for batches in results:
            for i, batch in enumerate(batches):
                model_name = self.model_flow.return_model_nodes[i].name
                all_results[model_name].append(batch)

        eval_info = {}
        if cur_iter is not None:
            eval_info["episode_iteration"] = cur_iter
        if train_iteration is not None:
            eval_info["train_iteration"] = train_iteration
        processed_results = self.post_process(all_results, eval_info)
        return processed_results

    def post_process(self, results, eval_info): # pylint: disable=unused-argument
        """
        Default post-process function for model evaluation results.

        Args
        ----
            results: list[]
                a list of evaluation results
            eval_info: dict[]
                a meta that contains "train_iteration" and "episode_iteration"
        """
        return results

    def get_and_clear_metrics(self):
        if self._metric_list is None or len(self._metric_list) == 0:
            return self._metric_prefix, {}

        reduced_metrics = map_reduce_metrics(self._metric_list)
        self._metric_list = []
        return self._metric_prefix, reduced_metrics
# pylint: disable=not-callable
