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

import threading
from collections import defaultdict
from itertools import cycle
from ray.util.queue import Queue
import torch

from chatlearn.models.vllm_module_v2 import VLLMModuleV2
from chatlearn.runtime.model_flow import ModelFlow
from chatlearn.utils import future
from chatlearn.utils.constant import CHATLEARN_REGROUP_TAG, INDEX_TAG
from chatlearn.utils.constant import LOG_START
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from .utils import encode_data, decode_data
from .utils import FlowParser



def split_list(lst, n):
    assert len(lst) % n == 0, f"{len(lst)} % {n} != 0"
    k = len(lst) // n
    return [lst[i*k:(i+1)*k] for i in range(n)]


def split_along_batch(tensors, num_splits):
    res = [{} for _ in range(num_splits)]
    if tensors is None:
        return res
    for key in tensors.keys():
        to_batch = tensors[key]
        if isinstance(to_batch, torch.Tensor):
            batched = to_batch.chunk(num_splits)
        elif isinstance(to_batch, list):
            batched = split_list(to_batch, num_splits)
        else:
            raise Exception(f"unknown types key: {key} and {type(to_batch)} to split: {key} {tensors.keys()} {to_batch}")
        for idx, ele in enumerate(batched):
            res[idx][key] = ele
    return res


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
        self.model2iter = {}
        self.merged_buffer = defaultdict(dict)
        self._metric_list = []

    def set_timers(self, _timers):
        self._timers = _timers

    @property
    def timers(self):
        return self._timers

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
        self.model_to_call_funcs = FlowParser().parse(flow)
        for model, func_names in self.model_to_call_funcs.items():
            model.call_funcs += func_names
        self.models = list(self.model_to_call_funcs.keys())
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
        self.model_locks = {model_node: threading.Lock() for model_node in self.model_flow.model_nodes}

    def _next_model(self, model):
        if len(model.replicas) == 1:
            return model.replicas[0]
        if model not in self.model2iter:
            self.model2iter[model] = cycle(iter(model.replicas))
        return next(self.model2iter[model])

    def get_merged_data(self, queues, encode=True, micro_batch_index=None, model_node=None, trainable=False):
        mb0 = None
        if micro_batch_index is not None:
            mb0 = micro_batch_index
        data_list = [None] * len(queues)
        merged_buffer = self.merged_buffer[model_node]
        for index, queue in enumerate(queues):
            if index not in merged_buffer:
                merged_buffer[index] = {}
            if mb0 in merged_buffer[index]:
                data_list[index] = merged_buffer[index].pop(mb0)
                continue
            while True:
                flag = False
                while queue.qsize() == 0:
                    if mb0 in merged_buffer[index]:
                        data_list[index] = merged_buffer[index].pop(mb0)
                        flag = True
                        break
                if flag:
                    break
                encoded_data = queue.get()
                mb, data = decode_data(encoded_data)
                if mb0 is None:
                    mb0 = mb
                if isinstance(data, list) and not trainable:
                    data = data[-1]
                if mb == mb0:
                    data_list[index] = data
                    break
                merged_buffer[index][mb] = data
        if encode:
            return encode_data(mb0, data_list)
        return data_list

    def get_merged_data_locked(self, queues, encode=True, micro_batch_index=None, model_node=None, trainable=False):
        with self.model_locks[model_node]:
            return self.get_merged_data(queues, encode, micro_batch_index, model_node, trainable)

    @staticmethod
    def align_out_queues(queues, encode=False):
        # TODO: deal with one2many scene
        out_queues = []
        min_qsize = min([ele.qsize() for ele in queues]) # pylint: disable=consider-using-generator
        for queue in queues:
            num_producers = queue.qsize()
            if num_producers == min_qsize:
                out_queues.append(queue)
                continue
            assert num_producers % min_qsize == 0
            out_queue = Queue()
            res_list = []
            while queue.qsize() > 0:
                res = queue.get()
                res = decode_data(res)[1] if encode else res
                res_list.append(res)

            division = num_producers // min_qsize
            in_qsize = len(res_list)
            out_qsize = in_qsize // division
            for q_idx in range(out_qsize):
                start = q_idx * division
                end = start + division
                out_queue.put(encode_data(q_idx, {CHATLEARN_REGROUP_TAG:res_list[start:end]}))
            out_queues.append(out_queue)
        return out_queues

    def get_all_merged_data(self, queues, out_queue, encode=True):
        logger.info(f"{LOG_START} start to align output queues with sizes {[ele.qsize() for ele in queues]}.")
        queues = self.align_out_queues(queues, True)
        logger.info(f"{LOG_START} complete to align output queues, sizes of output_queues are {[ele.qsize() for ele in queues]}.")
        queue0 = queues[0]
        while queue0.qsize() > 0:
            res = self.get_merged_data(queues, encode)
            out_queue.put(res)

    def rebatch_all_merged_data(self, model_node, in_queues, is_eval=False):# pylint: disable=unused-argument
        if not model_node.input_nodes:
            return in_queues
        out_queues = [None] * len(in_queues)
        num_consumers = self.batch_per_episode(model_node.model)
        for index, (input_node, in_queue) in enumerate(zip(model_node.input_nodes, in_queues)):
            num_producers = self.batch_per_episode(input_node.model)
            if num_producers == num_consumers:
                out_queues[index] = in_queue
            else:
                out_queues[index] = Queue()
                res_list = []
                while in_queue.qsize() > 0:
                    res = in_queue.get()
                    res = decode_data(res)[1]
                    res_list.append(res)
                if num_producers > num_consumers:
                # Deal with the case where num_producers > num_consumers
                    assert num_producers % num_consumers == 0, \
                        f"many2one: num_producers: {num_producers}, num_consumers: {num_consumers}, len inqueue: {len(in_queues)}"
                    division = num_producers // num_consumers
                    in_qsize = len(res_list)
                    out_qsize = in_qsize // division
                    for q_idx in range(out_qsize):
                        start = q_idx * division
                        end = start + division
                        out_queues[index].put(encode_data(q_idx, {CHATLEARN_REGROUP_TAG:res_list[start:end]}))
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
                        out_queues[index].put(encode_data(q_idx, {CHATLEARN_REGROUP_TAG: res_list[start:end],
                                                              INDEX_TAG: (q_idx % division, division)}))
        return out_queues

    def get_next_data(self, in_queue, model_node, micro_batch_index):
        if isinstance(in_queue, list):
            if len(in_queue) > 0:
                # this should happen for inference models, will trigger bug for training models
                # since training models accept a list of remote object, which has the same
                # behavior for models accept multiple inputs
                # we need to deal with it later
                assert not model_node.trainable
                data = self.get_merged_data_locked(in_queue, micro_batch_index=micro_batch_index,
                                                    model_node=model_node, trainable=model_node.trainable)
                mb, query = decode_data(data)
            else:
                mb, query = micro_batch_index, []
        else:
            data = self.get_merged_data_locked([in_queue], micro_batch_index=micro_batch_index,
                                                model_node=model_node, trainable=model_node.trainable)
            assert len(data['data']) == 1
            data['data'] = data['data'][0]
            mb, query = decode_data(data)
            query = [query]
        return mb, query

    def generate_step_one_model_internal(self, model_node, in_queue, step_num, replica, func_name="forward_step", to_empty_cache=None,
                                         is_eval=False, to_onload=None, to_offload=None, micro_batch_index=None):
        """
        Args:
            model: DistModel
            in_queue: Queue
            step_num: int
            replica: current model replica of DistModel
            func_name: str
            to_empty_cache: None or boolean
        """
        model = model_node.model
        kwargs = {}

        replica_num = len(model.replicas)
        output = []
        if isinstance(replica.model, VLLMModuleV2):
            last_step_start = max(self.num_iteration(model) - replica_num, 0)
            is_last_batch = step_num >= last_step_start
            kwargs["is_last_batch"] = is_last_batch
            if is_eval is not None:
                kwargs["is_eval"] = is_eval
            if to_empty_cache is not None:
                kwargs["to_empty_cache"] = to_empty_cache
            if to_onload is not None:
                kwargs["to_onload"] = to_onload
            if to_offload is not None:
                kwargs["to_offload"] = to_offload
            mb, query = self.get_next_data(in_queue, model_node, micro_batch_index)
            assert isinstance(query, list)
            # ret = replica.call_actor_remote_func(replica.vllm_engine, func_name, *query, **kwargs)
            ret = [[replica.call_actor_remote_func(engine, func_name, *query, **kwargs) for engine in replica.vllm_engines]]
            output.append((ret, mb))
        else:
            last_step_start = max(self.num_iteration(model) - replica_num, 0)
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
            for _, actors in replica.dp_rank_to_actors.items():
                mb, query = self.get_next_data(in_queue, model_node, micro_batch_index)
                assert isinstance(query, list)
                for actor in actors:
                    ret = replica.call_actor_remote_func(actor, func_name, *query, **kwargs)
                    output.append((ret, mb))
        return output

    def generate_step_one_model(self, model_node, replica, in_queue, out_queue, step_num, func_name="forward_step",
                                to_empty_cache=None, is_eval=False, to_onload=None, to_offload=None, micro_batch_index=None):
        """
        Args:
            model: DistModel
            in_queue: Queue
            out_queue: Queue
            step_num: int
            func_name: str
            to_empty_cache: None or boolean
        """
        model = model_node.model
        # output is a list of tuple, each tuple is (remote_refs, mb)
        output = self.generate_step_one_model_internal(model_node, in_queue, step_num, replica, func_name, to_empty_cache,
                                                       is_eval, to_onload, to_offload, micro_batch_index)

        num_dp_rank = len(replica.dp_rank_to_actors)
        if model.module_args.zero_size == 1:
            # If (tp > 1 or pp > 1) and ep = 1 for current model, its `output` will be a list whose
            #   length is the number of Actors. In this case, all members in the list
            #   are the same, and we choose output[-1] to put into out_queue.
            # If (tp > 1 or pp > 1) and ep > 1, we choose last output for each dp rank to put into
            #   out_queue.
            if model.module_args.expert_model_parallel_size == 1 and num_dp_rank == 1:
                result = [output[-1]]
            else:
                num_output = len(output)
                assert num_output % num_dp_rank == 0, (
                    f"The number of outputs ({num_output}) must be divisible by "
                    f"the number of dp_ranks ({num_dp_rank}) in a replica."
                )
                interval = num_output // num_dp_rank
                result = [output[i] for i in range(interval - 1, num_output, interval)]
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

    def regroup_inqueue(self, model_node, queues, is_eval=False):
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

    def compute_loop_one_model(self, model_node, num_batch=None):
        logger.info(f"{LOG_START} start compute_loop for {model_node}, is_eval={self.is_eval}")
        model = model_node.model
        is_eval = self.is_eval

        if num_batch is None:
            num_batch = self.num_iteration(model)

        func_name = model_node.func_name
        if model_node.remote_objects_to_wait:
            logger.info(f"{LOG_START} start to wait colocate models to finish for {model_node}")
            model_node.wait_colocate_models_to_finish(self.timers, func_name)
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
        self.timers(f"{model.name}").start()
        for step in range(num_batch):
            to_empty_cache = step >= last_step_start and (model.is_colocate or model.module_args.force_free_memory)
            to_onload = step < replica_num and ((model.is_colocate and model.enable_offload) or model.module_args.force_free_memory)
            to_offload = step >= last_step_start and ((model.is_colocate and model.enable_offload) or model.module_args.force_free_memory)
            replica = self._next_model(model)
            _, data = self.generate_step_one_model(model_node, replica, in_queue, model_node.out_queues, step, func_name, to_empty_cache,
                                                   is_eval=is_eval, to_onload=to_onload, to_offload=to_offload)
            results.append(data)
        self.timers(f"{model.name}").stop()
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

    def compute_loop(self, out_queue, num_batch=None):
        for model_group in self.model_flow.flow_topology:
            for model_node in model_group:
                self.compute_loop_one_model(model_node, num_batch)

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
            num_out_queue = len(model_node.output_nodes)
            if model_node in self.model_flow.return_model_nodes:
                num_out_queue += 1
            model_node.set_out_queues([Queue() for _ in range(num_out_queue)])
        return data_queues, out_queue
# pylint: disable=not-callable
