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
"""Model FLow"""

from collections import defaultdict, deque

from chatlearn.utils import future
from chatlearn.utils.utils import flatten


class DummyData:
    """DummyData to trace ModelGraph"""

    def __init__(self, from_model=None):
        self.from_model = from_model
        self.to_models = []


class ModelNode:
    """ModelNode"""

    def __init__(self, model, model_arg_name, func_name):
        self.model = model
        self.name = model.name
        self.model_arg_name = model_arg_name
        self.func_name = func_name
        self.input_models = []
        self.output_models = []
        self.out_queues = None
        self._input_queue = None
        # next colocate model node to execute
        self.next_colocate_node = None
        # model to wait before the execution of current model
        self.models_to_wait = []
        # remote objects to wait before the execution of current model
        self.remote_objects_to_wait = []

    def add_input_node(self, model):
        self.input_models.append(model)
        model.add_output_node(self)

    def add_output_node(self, model):
        self.output_models.append(model)

    def set_out_queues(self, queues):
        self.out_queues = queues

    def set_input_queue(self, queue):
        self._input_queue = queue

    def get_input_queues(self):
        input_queues = []
        if self._input_queue is not None:
            input_queues.append(self._input_queue)
        for input_model_node in self.input_models:
            out_index = input_model_node.output_models.index(self)
            input_queues.append(input_model_node.out_queues[out_index])
        if len(input_queues) == 1:
            return input_queues[0]
        return input_queues

    def _find_all_parents(self, model, prev_models_results):
        parents_models = []
        parents_results = []
        queue = deque([model])
        visited = set()
        while queue:
            cur_model = queue.pop()
            if cur_model in visited:
                continue
            visited.add(cur_model)
            for prev_model, results in prev_models_results:
                if prev_model in cur_model.input_models and prev_model not in parents_models:
                    parents_models.append(prev_model)
                    parents_results.append(results)
                    queue.append(prev_model)
        # reverse
        return parents_models[::-1], parents_results[::-1]


    def add_dependent_colocate_model_results(self, model, remote_objects, models_and_results_to_wait):
        # for models that are not colocated with current model, if their colocated model need to wait
        # the parent of their colocated model also need to wait
        dependent_models_not_colocate,  dependent_results_not_colocate = self._find_all_parents(model, models_and_results_to_wait)
        models_and_results_to_wait2 = [(model, results) for model, results in models_and_results_to_wait \
                                       if model not in dependent_models_not_colocate]
        for prev_model, result in zip(dependent_models_not_colocate, dependent_results_not_colocate):
            self.models_to_wait.append(prev_model)
            self.remote_objects_to_wait.extend(result)
        self.models_to_wait.append(model)
        self.remote_objects_to_wait.extend(remote_objects)
        return models_and_results_to_wait2

    def wait_colocate_models_to_finish(self, timers, func_name):
        for model in self.models_to_wait:
            timers(f"{model.name}").start()
        future.wait(self.remote_objects_to_wait, f"{[model.name for model in self.models_to_wait]} {func_name}")
        for model in self.models_to_wait:
            timers(f"{model.name}").stop()
        self.remote_objects_to_wait = []
        self.models_to_wait = []

    def __str__(self):
        return f"{self.__class__.__name__}({self.model})"

    def __repr__(self):
        return f'<{self.__class__.__name__}({self.model}) object at {hex(id(self))}>'


def fake_compute():
    def inner(self, *args):
        for data in args:
            if isinstance(data, DummyData):
                data.to_models.append(self)
                self._dummy_inputs.append(data)
        res = DummyData(self)
        self._dummy_output = res
        return res

    return inner


class ModelFlow:
    """ModelFlow"""

    def __init__(self, cls):
        self.model_nodes = []
        self.return_model_nodes = []
        self.out_to_model_node = {}
        self.cls = cls
        # models that consumes input data
        self.input_consumers = []

    def get(self, name):
        return self.name_to_node[name]

    def trace(self, models, compute_flow):
        """
        Trace the model compute_flow to get model graph.

        Args
        ----
        models: List(DistModel)
            a list of DistModel
        compute_flow: callable
            compute_flow function
        """
        local_models = [model.replicas[0].model for model in models]
        name2remote_model = {model.name: model for model in models}
        class_to_old_func = {}
        for model in local_models:
            func_name = self.cls.model_to_call_func[model]
            class_to_old_func[(model, func_name)] = getattr(model.__class__, func_name)
            setattr(model.__class__, func_name, fake_compute())

        dummy_data = DummyData()
        assert compute_flow is not None
        dummy_output = compute_flow(dummy_data)
        # convert decorator back
        for model in local_models:
            func_name = self.cls.model_to_call_func[model]
            setattr(model.__class__, func_name, class_to_old_func[(model, func_name)])

        for model in local_models:
            remote_model = name2remote_model[model.name]
            node = ModelNode(remote_model, model.name, self.cls.model_to_call_func[model])
            if model._dummy_output:
                self.out_to_model_node[model._dummy_output] = node
            for dummy_input in model._dummy_inputs:
                if dummy_input in self.out_to_model_node:
                    node.add_input_node(self.out_to_model_node[dummy_input])
            self.model_nodes.append(node)
        if dummy_output:
            if isinstance(dummy_output, DummyData):
                dummy_output = [dummy_output]
            for do in dummy_output:
                self.return_model_nodes.append(self.out_to_model_node[do])

        self.name_to_node = {node.model.name: node for node in self.model_nodes}
        self.input_consumers = [self.name_to_node[model.name] for model in dummy_data.to_models]
        self.flow_topology = self.topological_sort()
        self.model_nodes = flatten(self.flow_topology)
        for i, current_node in enumerate(self.model_nodes):
            for j in range(i + 1, len(self.model_nodes)):
                if not current_node.model.colocate_models:
                    break
                next_node = self.model_nodes[j]
                if current_node.model.colocate_with(next_node.model):
                    current_node.next_colocate_node = next_node
                    break
        # reset dummy info
        for model in local_models:
            model._dummy_inputs = []
            model._dummy_output = None

    def topological_sort(self):
        result = []
        level_map = defaultdict(list)
        in_degree = defaultdict(int)

        # Calculate the in-degree of each vertex
        for u in self.model_nodes:
            for v in u.output_models:
                in_degree[v] += 1

        # Enqueue all the vertices with an in-degree of 0
        queue = deque([u for u in self.model_nodes if in_degree[u] == 0])

        # Perform topological sorting
        while queue:
            current_level = []
            for _ in range(len(queue)):
                current = queue.popleft()
                current_level.append(current)
                result.append(current)

                # Decrement the in-degree of adjacent vertices
                for v in current.output_models:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

            level_map[len(result)].extend(current_level)

        # Check if the graph contains a cycle
        if len(result) != len(self.model_nodes):
            return None
        return [v[1] for v in sorted(level_map.items())]
