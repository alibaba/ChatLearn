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
"""Executor"""

from itertools import cycle
from ray.util.queue import Queue

from chatlearn.runtime.model_flow import ModelFlow, ModelNode
from chatlearn.utils import future
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from .utils import encode_data, decode_data
from .utils import FlowParser


# pylint: disable=not-callable
class Executor:
    """Executor"""

    def __init__(self, model_flow):
        """
        Executor

        Args
        ----
        models : List[BaseModule]
            a list of modules
        """
        self._set_flow(model_flow)
        self.args = get_args().runtime_args
        self.model_flow = None
        self.local_models = self.models
        self._batch_per_episode = -1
        self.is_eval = False
        self._timers = None

    def set_timers(self, _timers):
        self._timers = _timers

    @property
    def timers(self):
        return self._timers

    @property
    def batch_per_episode(self):
        return self._batch_per_episode

    def _set_flow(self, flow):
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
        self.model_to_call_func = FlowParser().parse(flow)
        for model, func_name in self.model_to_call_func.items():
            model.call_funcs.append(func_name)
        self.models = list(self.model_to_call_func.keys())
        return self

    @property
    def first_node(self):
        return self.model_flow.model_nodes[0]

    @property
    def first_model(self):
        return self.first_node.model

    def update_models(self, models):
        # update local model with remote models
        new_models = []
        name_to_new_models = {model.name: model for model in models}
        for model in self.local_models:
            dist_model = name_to_new_models[model.name]
            dist_model.group_dist_actors_by_tp_rank()
            new_models.append(dist_model)
        self.models = new_models
        if self.args is None:
            self.args = get_args().runtime_args

    def setup(self):
        self._models_and_results_to_wait = []
        self.model_flow = ModelFlow(self)
        self.model_flow.trace(self.models, self._flow)
        self.models = [model_node.model for model_node in self.model_flow.model_nodes]

    def _get_model(self, model):
        if len(model.replicas) == 1:
            return model.replicas[0]
        if model not in self.model2iter:
            self.model2iter[model] = cycle(iter(model.replicas))
        return next(self.model2iter[model])

    def get_merged_data(self, queues, encode=True):
        queue0 = queues[0]

        mb0, data0 = decode_data(queue0.get())
        if isinstance(data0, list):
            # if model has multiple actors, if TP>1/PP>1, just use the last
            data0 = data0[-1]
        data_list = [None] * len(queues)
        data_list[0] = data0
        for index, queue in enumerate(queues[1:]):
            if index not in self.merged_buffer:
                self.merged_buffer[index] = {}
            if mb0 in self.merged_buffer[index]:
                data_list[index+1] = self.merged_buffer[index].pop(mb0)
                continue
            while True:
                encoded_data = queue.get()
                mb, data = decode_data(encoded_data)
                if isinstance(data, list):
                    data = data[-1]
                if mb == mb0:
                    data_list[index+1] = data
                    break
                self.merged_buffer[index][mb] = data
        if encode:
            return encode_data(mb0, data_list)
        return data_list

    def get_all_merged_data(self, queues, out_queue, encode=True):
        queue0 = queues[0]
        while queue0.qsize() > 0:
            res = self.get_merged_data(queues, encode)
            out_queue.put(res)

    def execute_onload(self, model_node):
        if isinstance(model_node, ModelNode):
            model = model_node.model
        else:
            model = model_node
        # TODO: overlap with execution of other models
        refs = model.onload()
        future.wait(refs)

    def execute_offload(self, model_node):
        if isinstance(model_node, ModelNode):
            model = model_node.model
        else:
            model = model_node
        refs = model.offload()
        future.wait(refs)

    def generate_step_one_model_internal(self, model, in_queue, step_num, replica=None, func_name="forward_step", to_empty_cache=None,
                                         is_eval=False, to_onload=None, to_offload=None):
        """
        Args:
            model: DistModel
            in_queue: Queue
            step_num: int
            replica: current model replica of DistModel
            func_name: str
            to_empty_cache: None or boolean
        """
        if replica is None:
            replica = self._get_model(model)

        def get_next_data():
            if isinstance(in_queue, list):
                # this should happen for inference models, will trigger bug for training models
                # since training models accept a list of remote object, which has the same
                # behavior for models accept multiple inputs
                # we need to deal with it later
                assert not model.trainable
                data = self.get_merged_data(in_queue)
                mb, query = decode_data(data)
            else:
                data = in_queue.get()
                mb, query = decode_data(data)
                query = [query]
            return mb, query
        kwargs = {}

        replica_num = len(model.replicas)
        last_step_start = max(self.batch_per_episode - replica_num, 0)
        is_last_batch = step_num >= last_step_start
        kwargs["is_last_batch"] = is_last_batch
        if to_empty_cache is not None:
            kwargs["to_empty_cache"] = to_empty_cache
        if to_onload is not None:
            kwargs["to_onload"] = to_onload
        if to_offload is not None:
            kwargs["to_offload"] = to_offload
        if is_eval is not None:
            kwargs["is_eval"] = is_eval
        output = []
        for _, actors in replica.dp_rank_to_actors.items():
            mb, query = get_next_data()
            assert isinstance(query, list)
            for actor in actors:
                ret = replica.call_actor_remote_func(actor, func_name, *query, **kwargs)
                output.append((ret, mb))
        return output

    def generate_step_one_model(self, model, in_queue, out_queue, step_num, func_name="forward_step",
                                to_empty_cache=None, is_eval=False, to_onload=None, to_offload=None):
        """
        Args:
            model: DistModel
            in_queue: Queue
            out_queue: Queue
            step_num: int
            func_name: str
            to_empty_cache: None or boolean
        """
        # output is a list of tuple, each tuple is (remote_refs, mb)
        output = self.generate_step_one_model_internal(model, in_queue, step_num, None, func_name, to_empty_cache,
                                                       is_eval, to_onload, to_offload)

        # If tp > 1 or pp > 1 for current model, its `output` will be a list whose
        #   length is the number of Actors. In this case, all members in the list
        #   are the same, and we choose output[-1] to put into out_queue.
        if model.module_args.zero_size == 1:
            result = [output[-1]]
        else:
            result = output
        if isinstance(out_queue, list):
            for oq in out_queue:
                for res, mb in result:
                    oq.put(encode_data(mb, res))
        else:
            for res, mb in result:
                out_queue.put(encode_data(mb, res))
        # To ensure all Actors are finished synchronously, all remote refs should be returned
        # note that ray wait does not support tuple type, return a list of list
        remote_refs = [item[0] for item in output]
        return out_queue, remote_refs

    def compute_loop_one_model(self, model_node, num_batch, is_eval):
        model = model_node.model

        func_name = model_node.func_name
        if model_node.remote_objects_to_wait:
            model_node.wait_colocate_models_to_finish(self.timers, func_name)
        replica_num = len(model.replicas)
        last_step_start = max(num_batch - replica_num, 0)
        in_queue = model_node.get_input_queues()
        out_queue = model_node.out_queues
        results = []
        self.timers(f"{model.name}").start()
        for step in range(num_batch):
            to_empty_cache = step >= last_step_start and model.is_colocate
            to_onload = step < replica_num and model.is_colocate and model.enable_offload
            to_offload = step >= last_step_start and model.is_colocate and model.enable_offload
            _, data = self.generate_step_one_model(model, in_queue, out_queue, step, func_name, to_empty_cache,
                                                   is_eval=is_eval, to_onload=to_onload, to_offload=to_offload)
            results.append(data)
        self.timers(f"{model.name}").stop()
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
            logger.info(f"Sync {model} in the end of {self.__class__.__name__}")
            self._models_and_results_to_wait.append((model_node, results))

        return results

    def compute_loop(self, out_queue, num_batch):
        for model_group in self.model_flow.flow_topology:
            for model_node in model_group:
                self.compute_loop_one_model(model_node, num_batch, self.is_eval)

        data = [None] * len(self.model_flow.return_model_nodes)
        for model_node in self.model_flow.model_nodes:
            self.timers(f"{model_node.model.name}").start()
            if model_node in self.model_flow.return_model_nodes:
                # let the results order follow model_node order
                data[self.model_flow.return_model_nodes.index(model_node)] = model_node.out_queues[-1]
            self.timers(f"{model_node.model.name}").stop()
        model_names = []
        results = []
        for model, result in self._models_and_results_to_wait:
            model_names.append(model.name)
            results.extend(result)
        if results:
            for model_name in model_names:
                self.timers(f"{model_name}").start()
            func_name = self.model_flow.model_nodes[0].func_name
            future.wait(results, f"{model_names} {func_name}")
            for model_name in model_names:
                self.timers(f"{model_name}").stop()
            self._models_and_results_to_wait = []
        if data:
            self.get_all_merged_data(data, out_queue, encode=False)

    def setup_queues(self):
        data_queues = []
        out_queue = Queue()
        for model_node in self.model_flow.input_consumers:
            data_queue = Queue()
            data_queues.append(data_queue)
            model_node.set_input_queue(data_queue)
        for model_node in self.model_flow.model_nodes:
            num_out_queue = len(model_node.output_models)
            if model_node in self.model_flow.return_model_nodes:
                num_out_queue += 1
            model_node.set_out_queues([Queue() for _ in range(num_out_queue)])
        return data_queues, out_queue
# pylint: disable=not-callable
