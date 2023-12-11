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
"""Evaluator"""

import math
from itertools import cycle

from chatlearn.runtime.environment import Environment
from chatlearn.utils import future
from chatlearn.utils.logger import logger
from chatlearn.data.ranking import batch_generation_ranking


class Evaluator(Environment):
    """
    Evaluator.

    Args
    ----
    models : [RLHFModule]
        models to evaluate
    args : RLHFConfig
        default to None
    """

    def __init__(self, models):
        super().__init__(models)
        self._post_process_func = None
        self.is_eval = True

    @property
    def sample_per_episode(self):
        return len(self._dataset)

    @property
    def batch_per_episode(self):
        if self._batch_per_episode is not None:
            return self._batch_per_episode
        self._batch_per_episode = math.ceil(len(self._dataset) / self.batch_size)
        return self._batch_per_episode

    def setup_dataset(self):
        assert len(self._dataset) > 0, "dataset is not set"
        if self.models[0].module_args.batch_generation.ranking:
            logger.info("calling batch_generation_ranking")
            self._dataset = batch_generation_ranking(self._dataset, 1, len(dataset))
        refs = []
        for model_replica in self.models[0].replicas:
            ref = model_replica.master._build_dataloader.remote(self._dataset, self.batch_size, is_eval=True)
            refs.append(ref)
        future.get(refs)

    def eval_step(self, data_queue, out_queue, step):
        in_queue = data_queue
        for node in self.model_flow.model_nodes:
            model = node.model
            func_name = model.replicas[0].eval_call_func
            assert func_name is not None, \
                f"call model.register_eval_func for {model.name} before initializing Evaluator."
            out_queue = node.out_queues
            self.generate_step_one_model(model, in_queue, out_queue, step, func_name, False, is_eval=True)
            in_queue = node.out_queues[0]

        return self.get_merged_data(out_queue, encode=False)

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

    def eval(self, ppo_iter=None, train_iteration=None, return_last=True):
        """
        Evaluating.

        Args
        ----
        ppo_iter : int
            current ppo iteration.
        train_iteration: int
            current training iteration.
        return_last : bool
            return results of last model only.
        """
        num_batch = self.batch_per_episode
        refs = []
        for model in self.models[0].replicas:
            refs.append(model.master.reset_eval_data_iter.remote())
        future.get(refs)
        data_queues, out_queue = self.setup_queues()

        if self.use_vllm_backend:
            data_queue = data_queues[0]
            results = []
            for mb in range(num_batch):
                # add requests of current episode to vllm scheduler
                self.add_request(is_eval=True)

                # eval loop of current episode
                num_remaining_request = True
                while num_remaining_request:
                    step_output_rets = []
                    for model_replica in self.models[0].replicas:
                        query = model_replica.tailer.schedule.remote()
                        data_queue.put(self.encode_data(mb, query))
                        data = self.eval_step(data_queue, out_queue, mb)
                        step_output_rets.append(data)
                    num_remaining_request = future.get(step_output_rets)[0][0]

                # post precess of results in current episode
                outputs = []
                for model_replica in self.models[0].replicas:
                    outputs.append(self.vllm_post_process_outputs(model_replica))
                results += future.get(outputs)
        else:
            data_producer_iter = cycle(iter(self.models[0].replicas))
            for mb in range(num_batch):
                current_data_producer = next(data_producer_iter)
                query = current_data_producer.master.next_batch.remote(is_eval=True)
                encoded_data = self.encode_data(mb, query)
                for data_queue in data_queues:
                    data_queue.put(encoded_data)
            self.compute_loop(data_queues, out_queue, num_batch)
            queue_size = out_queue.qsize()
            result_refs = [out_queue.get() for _ in range(queue_size)]
            element_size = len(result_refs[0])
            results = future.wait(result_refs, desc="evaluator", return_output=True)
            results_nested = []
            for i in range(0, len(results), element_size):
                sublist = results[i:i+element_size]
                results_nested.append(sublist)
            results = results_nested
            if return_last:
                results = [res[0] for res in results]
            if self._post_process_func is not None:
                eval_info = {}
                if ppo_iter is not None:
                    eval_info["episode_iteration"] = ppo_iter
                if train_iteration is not None:
                    eval_info["train_iteration"] = train_iteration
                self._post_process_func(results, eval_info)
        return results
