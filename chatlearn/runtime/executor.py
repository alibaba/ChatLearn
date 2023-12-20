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
"""Executor"""

from itertools import cycle

from ray.util.queue import Queue

from chatlearn.runtime.model_flow import ModelFlow
from chatlearn.utils import future
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger


class Executor:
    """Executor"""

    def __init__(self, models):
        """
        Executor

        Args
        ----
        models : List[RLHFModule]
            a list of modules
        """
        if not isinstance(models, list):
            models = [models]
        self.args = get_args().rlhf_args
        self.model_flow = None
        self.models = models
        self.local_models = models
        self._batch_per_episode = -1
        self._flow = None
        self.is_eval = False

    @property
    def batch_per_episode(self):
        return self._batch_per_episode

    def set_flow(self, flow):
        """
        Set compution flow

        Args
        ----
        flow : callable
             a function that defines model computation flow

        Returns
        -------
        Executor
            return self
        """
        self._flow = flow
        return self

    def update_models(self, models):
        # update local model with remote models
        new_models = []
        name_to_new_models = {model.name: model for model in models}
        for model in self.local_models:
            new_models.append(name_to_new_models[model.name])
        self.models = new_models
        if self.args is None:
            self.args = get_args().rlhf_args

    def setup(self):
        self._models_and_results_to_wait = []
        self.model_flow = ModelFlow(self)
        self.model_flow.trace(self.models, self._flow, is_eval=self.is_eval)
        self.models = [model_node.model for model_node in self.model_flow.model_nodes]

    def encode_data(self, mb, data):
        return {"iter": mb, "data": data}

    def decode_data(self, data):
        mb = data["iter"]
        data = data["data"]
        return mb, data

    def _get_model(self, model):
        if len(model.replicas) == 1:
            return model.replicas[0]
        if model not in self.model2iter:
            self.model2iter[model] = cycle(iter(model.replicas))
        return next(self.model2iter[model])

    def get_merged_data(self, queues, encode=True):
        queue0 = queues[0]

        mb0, data0 = self.decode_data(queue0.get())
        if isinstance(data0, list):
            # if model has multiple actors, just use the last one
            data0 = data0[-1]
        data_list = [data0]
        for index, queue in enumerate(queues[1:]):
            if index not in self.merged_buffer:
                self.merged_buffer[index] = {}
            if mb0 in self.merged_buffer[index]:
                data_list.append(self.merged_buffer[index].pop(mb0))
                continue
            while True:
                encoded_data = queue.get()
                mb, data = self.decode_data(encoded_data)
                if isinstance(data, list):
                    data = data[-1]
                if mb == mb0:
                    data_list.append(data)
                    break
                self.merged_buffer[index][mb] = data
        if encode:
            return self.encode_data(mb0, data_list)
        return data_list

    def get_all_merged_data(self, queues, out_queue, encode=True):
        queue0 = queues[0]
        while queue0.qsize() > 0:
            res = self.get_merged_data(queues, encode)
            out_queue.put(res)

    def generate_step_one_model_internal(self, model, in_queue, step_num, func_name="forward_step", to_empty_cache=None,
                                         is_eval=False):
        """
        Args:
            model: DistModel
            in_queue: Queue
            step_num: int
            func_name: str
            to_empty_cache: None or boolean
        """
        is_train = model.trainable
        replica = self._get_model(model)

        if isinstance(in_queue, list):
            data = self.get_merged_data(in_queue)
            mb, query = self.decode_data(data)
        else:
            data = in_queue.get()
            mb, query = self.decode_data(data)
            query = [query]
        func = getattr(replica, func_name)
        kwargs = {}

        # if isinstance(query, list):
        replica_num = len(model.replicas)
        last_step_start = max(self.batch_per_episode - replica_num, 0)
        is_last_batch = step_num >= last_step_start
        kwargs["is_last_batch"] = is_last_batch
        if to_empty_cache is not None:
            kwargs["to_empty_cache"] = to_empty_cache
        if is_eval is not None:
            kwargs["is_eval"] = is_eval
        if is_train:
            kwargs["train_info"] = {"iteration": step_num}
        output = func(*query, **kwargs)
        return output, mb

    def generate_step_one_model(self, model, in_queue, out_queue, step_num, func_name="forward_step",
                                to_empty_cache=None, is_eval=False):
        """
        Args:
            model: DistModel
            in_queue: Queue
            out_queue: Queue
            step_num: int
            func_name: str
            to_empty_cache: None or boolean
        """
        output, mb = self.generate_step_one_model_internal(model, in_queue, step_num, func_name, to_empty_cache,
                                                           is_eval)

        # If tp > 1 or pp > 1 for current model, its `output` will be a list whose
        #   length is the number of Actors. In this case, all members in the list
        #   are the same, and we choose output[-1] to put into out_queue.
        last_output = output[-1] if isinstance(output, list) else output
        if isinstance(out_queue, list):
            for oq in out_queue:
                oq.put(self.encode_data(mb, last_output))
        else:
            out_queue.put(self.encode_data(mb, last_output))
        # To ensure all Actors are finished synchronously, `output` itself should be returned
        return out_queue, output

    def compute_loop_one_model(self, model_node, num_batch, is_eval):
        model = model_node.model
        # TODO: overlap with execution of other models
        if model.trainable and model.module_args.offload_optimizer_states:
            refs = model.onload_optimizer_states()
            future.wait(refs)
        func_name = model_node.func_name
        if model_node.remote_objects_to_wait:
            model_node.wait_colocate_models_to_finish(func_name)
        replica_num = len(model.replicas)
        last_step_start = max(num_batch - replica_num, 0)
        in_queue = model_node.get_input_queues()
        out_queue = model_node.out_queues
        results = []
        for step in range(num_batch):
            to_empty_cache = step >= last_step_start and model.need_empty_cache
            _, data = self.generate_step_one_model(model, in_queue, out_queue, step, func_name, to_empty_cache,
                                                   is_eval=is_eval)
            results.append(data)
        if model_node.next_colocate_node:
            # before the execution of next colocate model, perform the wait, since we want to empty the cache.
            logger.info(
                f"Model {model_node.next_colocate_node} will wait model {model} to finish since they are colocated")
            self._models_and_results_to_wait = model_node.next_colocate_node.add_dependent_colocate_model_results(
                model_node, results, self._models_and_results_to_wait)
        elif model.colocate_models or model.trainable:
            # 1. the model may colocate with training/inference, so we should wait until the end of compute_loop
            # 2. the model is trainable and it does not have next_colocate_model, we should make sure it is finished before parameter_sync
            # so we add them to a temp list
            logger.info(f"Sync {model} in the end of {self}")
            self._models_and_results_to_wait.append((model_node, results))
        # TODO: overlap with execution of other models
        if model.trainable and model.module_args.offload_optimizer_states:
            refs = model.offload_optimizer_states()
            future.wait(refs)
        return results

    def compute_loop(self, data_queues, out_queue, num_batch):
        for i, model_group in enumerate(self.model_flow.flow_topology):
            if i == 0:
                for j, model_node in enumerate(model_group):
                    model_node.set_input_queues(data_queues[j])
            for model_node in model_group:
                self.compute_loop_one_model(model_node, num_batch, self.is_eval)
        data = []
        for model_node in self.model_flow.model_nodes:
            if model_node in self.model_flow.return_model_nodes:
                data.append(model_node.out_queues[-1])
        model_names = []
        results = []
        for model, result in self._models_and_results_to_wait:
            model_names.append(model.name)
            results.extend(result)
        if results:
            func_name = self.model_flow.model_nodes[0].func_name
            future.wait(results, f"{model_names} {func_name}")
            self._models_and_results_to_wait = []
        if data:
            self.get_all_merged_data(data, out_queue, encode=False)

    def setup_queues(self):
        data_queues = []
        out_queue = Queue()
        for i, model_group in enumerate(self.model_flow.flow_topology):
            if i == 0:
                for model_node in model_group:
                    data_queue = Queue()
                    data_queues.append(data_queue)
                    model_node.set_input_queues(data_queue)
            for model_node in self.model_flow.model_nodes:
                num_out_queue = len(model_node.output_models)
                if model_node in self.model_flow.return_model_nodes:
                    num_out_queue += 1
                model_node.add_out_queues([Queue() for i in range(num_out_queue)])
        return data_queues, out_queue
