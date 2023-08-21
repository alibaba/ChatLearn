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

from ray.util.queue import Queue

from chatlearn.models.rlhf_module import RLHFModule
from chatlearn.runtime.environment import PPOEnv
from chatlearn.utils import future
from chatlearn.utils import utils
from chatlearn.utils.logger import logger
from chatlearn.utils.global_vars import get_args


class Evaluator(PPOEnv):
    """
    Evaluator.

    Args
    ----
    models : [RLHFModule]
        models to evaluate
    args : RLHFConfig
        default to None
    """

    def __init__(self, models, args=None): # pylint: disable=super-init-not-called
        self._lazy_init = False
        self.args = args
        if not isinstance(models, list):
            models = [models]
        self.models = models
        self._dataset = []
        self.data_iter = None
        self._padding_config = {}
        self._name2models = {}
        self.merged_buffer = {}
        self.model2iter = {}
        self._original_dataset = None
        self._post_process_func = None
        self._evaluate_models_colocate = True

    @property
    def batch_size(self):
        return self.models[0].module_args.generation_batch_size

    def update_models(self, models):
        new_models = []
        name_to_new_models = {model.name: model for model in models}
        for model in self.models:
            new_models.append(name_to_new_models[model.name])
        self.models = new_models
        if self.args is None:
            self.args = get_args().rlhf_args

    def setup(self, model_packs=None):
        if self._lazy_init:
            self.set_dataset(self._original_dataset)

        refs = []
        for i, model_replica in enumerate(self.models[0].replicas):
            ref = model_replica.master._build_dataloader.remote(self._dataset[i], is_eval=True)
            refs.append(ref)
        future.get(refs)

        for model in self.models:
            config = future.get(model.replicas[0].master.padding_config.remote())
            self._padding_config.update(config)
            self._name2models[model.name] = model

        model_names = set({eval_model.name for eval_model in self.models})
        for model_pack in model_packs:
            model_name_pack = [model.name for model in model_pack]
            if len(model_name_pack) > 1 and model_names.issubset(model_name_pack):
                self._evaluate_models_colocate = False

    def batch_generation_ranking(self, in_data):
        def sort_fun(ele):
            chinese = ""
            others = ""
            for s in ele:
                if '\u4e00' <= s <= '\u9fa5':
                    chinese += s
                else:
                    others += s
            return len(others.split(" ")) + len(chinese)
        in_data.sort(key=sort_fun, reverse=True)
        return in_data

    def set_dataset(self, dataset): # pylint: disable=arguments-differ
        """
        set dataset.

        Args
        ----
        dataset : [str]
            a list of str
        """
        if isinstance(self.models[0], RLHFModule):
            self._original_dataset = dataset
            self._lazy_init = True
            return self
        data_len = len(dataset)
        data_part_num = self.models[0].num_replica
        if get_args().active_module_args.batch_generation.ranking:
            logger.info("calling batch_generation_ranking")
            dataset = self.batch_generation_ranking(dataset)
            self._dataset = [[] for _ in range(data_part_num)]
            num_batch_per_episode = math.ceil(data_len / self.batch_size)
            for batch in range(num_batch_per_episode):
                start = batch * self.batch_size
                end = min(start + self.batch_size, data_len)
                self._dataset[int(batch % data_part_num)] += dataset[start:end]
        else:
            indices = utils.split_index(data_len, data_part_num)

            for start, end in indices:
                data_part = dataset[start:end]
                self._dataset.append(data_part)
        return self

    @property
    def num_eval_iteration(self):
        return sum(math.ceil(len(data) / self.batch_size) for data in self._dataset)

    def eval_step(self, data_queue, model_out_queues, return_last=True):
        in_queue = data_queue
        for model in self.models:
            func_name = model.replicas[0].eval_func_name
            assert func_name is not None, \
                f"call model.register_eval_func for {model.name} before initializing Evaluator."
            self.generate_step_one_model(model, in_queue, model_out_queues[model], func_name, False)
            in_queue = model_out_queues[model][0]

        if return_last:
            out_queues = [model_out_queues[self.models[-1]][-1]]
        else:
            out_queues = [model_out_queues[model][-1] for model in self.models]
        return self.get_merged_data(out_queues, encode=False)

    def get_all_merged_data_list(self, queues, encode=True):
        queue0 = queues[0]
        merged_data_list = []
        while queue0.qsize() > 0:
            res = self.get_merged_data(queues, encode)
            merged_data_list.append(res)
        return merged_data_list

    def eval_loop_sync(self, data_queue, model_out_queues, num_batch, return_last=True):
        in_queue = data_queue
        for model in self.models:
            func_name = model.replicas[0].eval_func_name
            assert func_name is not None, \
                f"call model.register_eval_func for {model.name} before initializing Evaluator."
            self.eval_loop_one_model(model, in_queue, model_out_queues[model], func_name, num_batch)
            in_queue = model_out_queues[model][0]
        if return_last:
            out_queues = [model_out_queues[self.models[-1]][-1]]
        else:
            out_queues = [model_out_queues[model][-1] for model in self.models]
        return self.get_all_merged_data_list(out_queues, encode=False)

    def eval_loop_one_model(self, model, in_queue, out_queue, func_name, num_batch):
        replica_num = len(model.replicas)
        last_step_start = max(num_batch - replica_num, 0)
        for step in range(num_batch):
            if step >= last_step_start:
                self.generate_step_one_model(model, in_queue, out_queue, func_name, True)
            else:
                self.generate_step_one_model(model, in_queue, out_queue, func_name, False)

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
        result_refs = []
        data_queue = Queue()
        num_batch = self.num_eval_iteration
        refs = []
        for model in self.models[0].replicas:
            refs.append(model.master.reset_eval_data_iter.remote())
        future.get(refs)
        data_providers = cycle(iter(self.models[0].replicas))
        out_queues = {}

        for k, model in enumerate(self.models):
            if k < len(self.models) - 1:
                queue_num = 2
            else:
                queue_num = 1
            out_queues[model] = [Queue() for i in range(queue_num)]

        if self._evaluate_models_colocate:
            for mb in range(num_batch):
                query = next(data_providers).master.next_batch.remote(is_eval=True)
                data_queue.put(self.encode_data(mb, query))
            result_refs = self.eval_loop_sync(data_queue, out_queues, num_batch)
            results = future.wait(result_refs, desc="evaluator", return_output=True)
        else:
            for mb in range(num_batch):
                query = next(data_providers).master.next_batch.remote(is_eval=True)
                data_queue.put(self.encode_data(mb, query))
                data = self.eval_step(data_queue, out_queues)
                result_refs.append(data)
            results = future.wait_and_empty_cache(
                self.models, result_refs, desc="evaluator", return_output=True
            )
        element_size = len(result_refs[0])
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
