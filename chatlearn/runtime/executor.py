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


from collections import defaultdict
from itertools import cycle
from typing import List, Callable, Optional, Union

from ray.util.queue import Queue

from chatlearn.models.base_module import BaseModule
from chatlearn.models.vllm_module import VLLMModule
from chatlearn.models.sglang_module import SGLangModule
from chatlearn.runtime.model_flow import ModelFlow, ModelNode
from chatlearn.runtime.dist_actor import DistModel, DistActor
from chatlearn.utils import future
from chatlearn.utils.constant import REF_LIST, INDEX_TAG, LOG_START
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.runtime.utils import encode_data, decode_data, FlowParser

# pylint: disable=not-callable
class Executor:
    """Executor"""

    def __init__(self, model_flow: Callable):
        """
        Executor
        """
        self._set_flow(model_flow)
        self.args = get_args().runtime_args
        self.model_flow = None
        self.local_models: List[BaseModule] = self.models
        self._batch_per_episode = -1
        self.is_eval = False
        self._timers = None
        self.model2iter = {}
        self.merged_buffer = defaultdict(dict)
        self._metric_list = []

    def set_timers(self, _timers):
        self._timers = _timers

    @property
    def timers(self):
        return self._timers

    def _set_flow(self, flow: Callable):
        """
        parse flow function to get BaseModule2forward_func map which is used in ModelFlow.trace
        Args:
            flow(callable): a function that defines model computation flow

        """
        self._flow: Callable = flow
        # model_to_call_funcs example:
        # {
        #     Module1: ["func_name1"],
        #     Module2: ["func_name2"],
        #     ...
        # }
        self.model_to_call_funcs: defaultdict[BaseModule, List[str]] = FlowParser().parse(flow)
        for model, func_names in self.model_to_call_funcs.items():
            # BaseModule.call_funcs: List[str]
            model.call_funcs += func_names
        self.models: List[BaseModule] = list(self.model_to_call_funcs.keys())

    @property
    def first_node(self) -> ModelNode:
        """
        get the first ModelNode in Executor
        """
        return self.model_flow.model_nodes[0]

    @property
    def first_model(self) -> DistModel:
        """
        get the DistModel in first node
        """
        return self.first_node.model

    def update_models(self, models: List[DistModel]) -> None:
        """
        set self.models by input models
        """

        # update local model with remote models
        new_models = []
        name_to_new_models = {model.name: model for model in models}
        for model in self.local_models:
            dist_model = name_to_new_models[model.name]
            dist_model.group_dist_actors_by_dp_rank()
            new_models.append(dist_model)
        self.models = new_models
        if self.args is None:
            self.args = get_args().runtime_args

    def setup(self):
        """
        setup model_flow and get DistModel in the flow
        """
        self._models_and_results_to_wait = []
        self.model_flow = ModelFlow(self)
        self.model_flow.trace(self.models, self._flow)
        self.models: List[DistModel] = [model_node.model for model_node in self.model_flow.model_nodes]

    def _next_model(self, model: DistModel) -> DistActor:
        """
        return the next DistActor of DistModel which is used for current execute
        """
        if model not in self.model2iter:
            self.model2iter[model] = cycle(iter(model.replicas))
        return next(self.model2iter[model])

    @staticmethod
    def align_out_queues(queues: List[Queue], encode=False) -> List[Queue]:
        """
        Merge every queue in queues to the min_qsize in queues.
        Warning: 
        1. In graph, the data is ObjectRef, we can not get the real data.
        2. This function is used at the end of executor, align every queue to have same data size. 
        3. As the data is ObjectRef, we just regroup ObjectRef in every list item, and actual merge data in !!!decorator.compute_decorator
        """
        out_queues = []
        min_qsize = min([ele.qsize() for ele in queues]) # pylint: disable=consider-using-generator
        for queue in queues:
            cur_qsize = queue.qsize()
            # if cur_qsize==min_qsize just pass this queue
            if cur_qsize == min_qsize:
                out_queues.append(queue)
            # if cur_qize>min_qsize merge this queue length to min_qsize
            elif cur_qsize > min_qsize:
                assert cur_qsize % min_qsize == 0
                out_queue = Queue()
                res_list = []
                while queue.qsize() > 0:
                    res = queue.get()
                    res = decode_data(res)[1] if encode else res
                    res_list.append(res)

                division = cur_qsize // min_qsize
                out_qsize = cur_qsize // division

                for q_idx in range(out_qsize):
                    start = q_idx * division
                    end = start + division
                    # TODO: Refacor this nested var
                    # !!!the data in out_queue is Dict[int, Dict[str, List[ObjectRef]]]]
                    out_queue.put(encode_data(q_idx, {REF_LIST:res_list[start:end]}))
                out_queues.append(out_queue)
        return out_queues

    def get_merged_data(self,
                        queues: List[Queue],
                        encode: bool = True,
                        micro_batch_index: Optional[int] = None, # pylint: disable-next=unused-argument
                        model_node: Optional[ModelNode] = None, # pylint: disable-next=unused-argument
                        trainable: bool = False
                        ):
        """
        merge data from different queues, get data from queues for current node by dp-wise.
        It will be executed in form of a for loop for dp size-times.
        Args:
            queues: a list of ray Queue, the length of queues means the number of current node. 
            the queues[0].qsize() is the global dp size of current node.
            encode: whether encode or not, it seems only when model_node is None, encode is False
            micro_batch_index: it seems always be None
            model_node: related to self.merged_buffer, but don't know where to use
        """
        assert micro_batch_index is None, "micro_batch_index should be None, will be deprecated and removed in future"
        # TODO：remove unused pramater model_node and trainable
        data_list = []
        mb0 = None
        for queue in queues:
            encoded_data = queue.get()
            mb, data = decode_data(encoded_data)
            data_list.append(data)
            mb0 = mb if mb0 is None else mb0

        return encode_data(mb0, data_list) if encode else data_list

    def get_all_merged_data(self, queues: List[Queue], out_queue: Queue, encode=True):
        """
        Merge different node Queues into one output queue, only used at the end of executor
        """
        logger.info(f"{LOG_START} start to align output queues with sizes {[ele.qsize() for ele in queues]}.")
        queues = self.align_out_queues(queues, True)
        logger.info(f"{LOG_START} complete to align output queues, sizes of output_queues are {[ele.qsize() for ele in queues]}.")
        while queues[0].qsize() > 0:
            res = self.get_merged_data(queues, encode)
            out_queue.put(res)

    def rebatch_all_merged_data(self, model_node: ModelNode, in_queues: List[Queue], is_eval=False) -> List[Queue]:# pylint: disable=unused-argument
        """
        re-construct input_queues[node_num, previous_node_global_dp_size] to output_queues[node_num, current_node_global_dp_size]
        warning: input_queues and output_queues are ray.util.queue.Queue, we can't get real data in executor.
        will actual merge data in !!!decorator.compute_decorator
        """
        # if this node is the first node, just pass the queues
        if not model_node.input_nodes:
            return in_queues
        out_queues = [None] * len(in_queues)
        # in environment, num_consumers is global dp size(replica_dp_size*dp_size)
        num_consumers = self.global_dp_size(model_node.model)
        for index, (input_node, in_queue) in enumerate(zip(model_node.input_nodes, in_queues)):
            num_producers = self.global_dp_size(input_node.model)
            if num_producers == num_consumers:
                out_queues[index] = in_queue
            else:
                out_queues[index] = Queue()
                # convert input queue to res_list
                res_list = []
                while in_queue.qsize() > 0:
                    res = in_queue.get()
                    res = decode_data(res)[1]
                    res_list.append(res)
                # Deal with the case where num_producers > num_consumers
                if num_producers > num_consumers:
                    assert num_producers % num_consumers == 0, \
                        f"many2one: num_producers: {num_producers}, num_consumers: {num_consumers}, len inqueue: {len(in_queues)}"
                    division = num_producers // num_consumers
                    in_qsize = len(res_list)
                    out_qsize = in_qsize // division
                    for q_idx in range(out_qsize):
                        start = q_idx * division
                        end = start + division
                        out_queues[index].put(encode_data(q_idx, {REF_LIST:res_list[start:end]}))
                else:
                    # Deal with the case where num_producers < num_consumers
                    # TODO: add index for one2many case
                    assert num_consumers % num_producers == 0, \
                        f"one2many: num_producers: {num_producers}, num_consumers: {num_consumers}, len inqueue: {len(in_queues)}"
                    division = num_consumers // num_producers
                    in_qsize = len(res_list)
                    out_qsize = in_qsize * division
                    for q_idx in range(out_qsize):
                        start = q_idx // division
                        end = start + 1
                        # q_idx means dp rank
                        out_queues[index].put(encode_data(q_idx, {REF_LIST: res_list[start:end],
                                                              INDEX_TAG: (q_idx % division, division)}))
        return out_queues

    def get_next_data(self, in_queue: Union[Queue, List[Queue]], model_node: ModelNode, micro_batch_index):
        """
        get a dp-rank data
        """
        if isinstance(in_queue, list):
            if len(in_queue) > 0:
                # this should happen for inference models, will trigger bug for training models
                # since training models accept a list of remote object, which has the same
                # behavior for models accept multiple inputs
                # we need to deal with it later
                assert not model_node.trainable
                data = self.get_merged_data(in_queue, micro_batch_index=micro_batch_index,
                                                    model_node=model_node, trainable=model_node.trainable)
                mb, query = decode_data(data)
            else:
                mb, query = micro_batch_index, []
        else:
            data = self.get_merged_data([in_queue], micro_batch_index=micro_batch_index,
                                                model_node=model_node, trainable=model_node.trainable)
            assert len(data['data']) == 1
            data['data'] = data['data'][0]
            mb, query = decode_data(data)
            query = [query]
        return mb, query

    def generate_step_one_model_internal(self, model_node, in_queue, step_num, replica, func_name="forward_step", to_empty_cache=None,
                                         is_eval=False, to_onload=None, to_offload=None, micro_batch_index=None):

        """
        forward for a model replica
        """
        model = model_node.model

        replica_num = len(model.replicas)
        output = []
        last_step_start = max(self.num_iteration(model) - replica_num, 0)
        is_last_batch = step_num >= last_step_start
        kwargs = {
            "is_last_batch": is_last_batch,
            "is_eval": is_eval,
            "to_empty_cache": to_empty_cache,
            "to_onload": to_onload,
            "to_offload": to_offload
        }

        if isinstance(replica.model, (VLLMModule, SGLangModule)):
            # for rollout we only to pass data to engine for every replica
            mb, query = self.get_next_data(in_queue, model_node, micro_batch_index)
            assert isinstance(query, list)
            ret = replica.call_actor_remote_func(replica.engine, func_name, *query, **kwargs)
            # output length is num replica
            output.append((ret, mb))
        else:
            for _, actors in replica.dp_rank_to_actors.items():
                mb, query = self.get_next_data(in_queue, model_node, micro_batch_index)
                assert isinstance(query, list)
                for actor in actors:
                    ret = replica.call_actor_remote_func(actor, func_name, *query, **kwargs)
                    # output length is num actor
                    output.append((ret, mb))
        return output

    def generate_step_one_model(self, model_node, replica, in_queue, out_queue, step_num, func_name="forward_step",
                                to_empty_cache=None, is_eval=False, to_onload=None, to_offload=None, micro_batch_index=None):
        """
        forward for a model replica, and only set the output of last rank in dp rank to out_queue
        """
        # output is a list of tuple, each tuple is (remote_refs, mb)
        output = self.generate_step_one_model_internal(model_node, in_queue, step_num, replica, func_name, to_empty_cache,
                                                       is_eval, to_onload, to_offload, micro_batch_index)

        # for get the data in last actor of a dp rank
        # replica.dp_rank_to_actors : Dict[int, List[Actor]]
        num_dp_rank = len(replica.dp_rank_to_actors) # numer of actors in a dp rank
        num_output = len(output)
        assert num_output % num_dp_rank == 0, (
            f"The number of outputs ({num_output}) must be divisible by "
            f"the number of dp_ranks ({num_dp_rank}) in a replica."
        )
        interval = num_output // num_dp_rank
        # For each dp rank, get the last actor's output
        result = [output[i] for i in range(interval - 1, num_output, interval)]

        # Put encoded remote_refs into out_queue to avoid execution
        if isinstance(out_queue, list):
            for oq in out_queue:
                for res, mb in result:
                    oq.put(encode_data(mb, res))
        else:
            for res, mb in result:
                out_queue.put(encode_data(mb, res))
        # Get all remote_refs from output, using this list to do ray execution
        remote_refs = [item[0] for item in output]
        return out_queue, remote_refs

    def regroup_inqueue(self, model_node: ModelNode, queues, is_eval=False):
        """
        re-construct input_queues[node_num, previous_node_global_dp_size] to output_queues[node_num, current_node_global_dp_size]
        """
        if self.args.policy_to_regroup_queue == "global_barrier":
            # barrier to regroup all queues of producer node
            if not isinstance(queues, list):
                queues = [queues]
            logger.info(f"{LOG_START} regroup_inqueue in_queue {model_node}:  {[ele.qsize() for ele in queues]}")
            out_queues = self.rebatch_all_merged_data(model_node, queues, is_eval=is_eval)
            logger.info(f"{LOG_START} regroup_inqueue out_queues {model_node}:  {[ele.qsize() for ele in out_queues]}")
            return out_queues
        else:
            raise RuntimeError(f"Unsupported policy_to_regroup_queue {self.args.policy_to_regroup_queue}.")

    def compute_loop_one_model(self, model_node: ModelNode, num_batch=None):
        """
        forward for all model replica in the model mode
        """
        logger.info(f"{LOG_START} start compute_loop for {model_node}, is_eval={self.is_eval}")
        model = model_node.model
        is_eval = self.is_eval

        if num_batch is None:
            num_batch = self.num_iteration(model)

        func_name = model_node.func_name
        if model_node.remote_objects_to_wait:
            logger.info(f"{LOG_START} start to wait colocate models to finish for {model_node}")
            model_node.wait_colocate_models_to_finish(func_name)
            logger.info(f"{LOG_START} complete to wait colocate models to finish for {model_node}")
        replica_num = len(model.replicas)
        last_step_start = max(num_batch - replica_num, 0)
        in_queue = model_node.get_input_queues()

        logger.info(f"{LOG_START} start to regroup in_queue for {model_node}")
        in_queue = self.regroup_inqueue(model_node, in_queue, is_eval=is_eval)
        logger.info(f"{LOG_START} complete to regroup in_queue for {model_node}")

        if isinstance(in_queue, list) and len(in_queue) == 1:
            in_queue = in_queue[0]
        results = []
        logger.info(f"{LOG_START} start to generate_step_one_model for {model_node}")
        for step in range(num_batch):
            to_empty_cache = step >= last_step_start and model.is_colocate
            to_onload = step < replica_num and (model.is_colocate and model.enable_offload)
            to_offload = step >= last_step_start and (model.is_colocate and model.enable_offload)
            replica = self._next_model(model)
            _, data = self.generate_step_one_model(model_node, replica, in_queue, model_node.out_queues, step, func_name, to_empty_cache,
                                                   is_eval=is_eval, to_onload=to_onload, to_offload=to_offload)
            results.append(data)
        if model_node.next_colocate_node:
            # before the execution of next colocate model, perform the wait, since we want to empty the cache.
            logger.info(
                f"{LOG_START} Model {model_node.next_colocate_node} will wait model {model} to finish since they are colocated")
            self._models_and_results_to_wait = model_node.next_colocate_node.add_dependent_colocate_model_results(
                model_node, results, self._models_and_results_to_wait)
        elif model.colocate_models or model.trainable:
            # 1. the model may colocate with training/inference, so we should wait until the end of compute_loop
            # 2. the model is trainable and it does not have next_colocate_model, we should make sure it is finished before parameter_sync
            # so we add them to a temp list
            logger.info(f"{LOG_START} Sync {model} in the end of {self.__class__.__name__}")
            self._models_and_results_to_wait.append((model_node, results))

    def compute_loop(self, out_queue: Queue, num_batch=None):
        """
        Main graph-executor.
        Following BFS, remote call all functions in the model_flow.
        For each node in model_flow, the remote call will wait for it's colocated models to finish.
        Since each node get ObjectRefs as input, we don't need to explicit wait for it's parent node to finish (ray.get will wait).
        """
        for model_group in self.model_flow.flow_topology:
            for model_node in model_group:
                self.compute_loop_one_model(model_node, num_batch)

        data = [None] * len(self.model_flow.return_model_nodes)
        for model_node in self.model_flow.model_nodes:
            if model_node in self.model_flow.return_model_nodes:
                # let the results order follow model_node order
                data[self.model_flow.return_model_nodes.index(model_node)] = model_node.out_queues[-1]
        model_names = []
        results = []
        for model, result in self._models_and_results_to_wait:
            model_names.append(model.name)
            results.extend(result)
        if results:
            func_name = self.model_flow.model_nodes[0].func_name
            future.wait(results, f"{model_names} {func_name}", True)
            self._models_and_results_to_wait = []
        if data:
            self.get_all_merged_data(data, out_queue, encode=False)

    def setup_queues(self):
        """
        setup queues for input_node and output queues for every model node
        if node is return_model_nodes, there will be an external queue for output
        """
        data_queues = []
        out_queue = Queue()
        for model_node in self.model_flow.input_consumers:
            data_queue = Queue()
            data_queues.append(data_queue)
            model_node.set_input_queue(data_queue)
        for model_node in self.model_flow.model_nodes:
            num_out_queue = len(model_node.output_nodes)
            if model_node in self.model_flow.return_model_nodes:
                num_out_queue += 1
            model_node.set_out_queues([Queue() for _ in range(num_out_queue)])
        return data_queues, out_queue
# pylint: disable=not-callable
