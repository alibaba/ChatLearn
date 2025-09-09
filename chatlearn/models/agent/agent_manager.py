# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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
"""agent manager"""
import ray
import uuid
from typing import Dict, List, Any
from collections import defaultdict
import itertools
import random

import torch
from transformers import AutoTokenizer

from chatlearn import BaseModule
from chatlearn.data.prompt_dataset import PromptPipeline
from chatlearn.runtime.decorator import monitor_error, compute_decorator, timeit

from typing import List, Dict, Iterator
from itertools import cycle


import heapq

# def data_balance(data: List[Dict], num_groups, group_size, length_key='all_token_length'):
def data_balance(data: List[Dict], dp_size, batch_size, key='all_token_length'):
    """
    balance total token budget between dp
    """
    sorted_data = sorted(data, key=lambda x: x[key], reverse=True)
    heap = [(0,i,[]) for i in range(dp_size)]
    heapq.heapify(heap)
    full_groups = []
    # Greedy
    for item in sorted_data:
        if not heap:
            break
        total_len, idx, group = heapq.heappop(heap)
        group.append(item)
        new_total_len = total_len + item[key]
        
        if len(group) >= batch_size:
            full_groups.append(group)
        else:
            heapq.heappush(heap, (new_total_len, idx, group))
    
    while heap:
        total_len, idx, group = heapq.heappop(heap)
        full_groups.append(group)

    result = []
    for group_data in full_groups:
        result += group_data
    
    return result

class AgentManager(BaseModule):
    """Agent Manager"""

    def __init__(self, name: str, args=None, replica_id: int=0):
        """ChatLearn main agent entrypoint
        """
        super().__init__(name, args=args, replica_id=replica_id)
        assert self.total_gpu == 0, "AgentManager does not require GPU"
        self._num_gpu_per_replica = 0
        self._num_replica = self.module_args.num_cpu // self.module_args.cpu_per_process
        assert self._num_replica == 1, "There should only be one AgentManager"

    @monitor_error()
    def setup(self):
        self.stats = {}
        self._metric_prefix = "agent"
        self.get_rollout()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.global_args.models.policy.load, trust_remote_code=True
        )


    def get_rollout(self, namespace="policy"):
        all_actors = ray.util.list_named_actors(all_namespaces=True)
        rollout_actors = [actor for actor in all_actors if actor['namespace'] == namespace]
        self.rollout_workers = [ray.get_actor(**item) for item in rollout_actors]
        # only support async sglang
        self.rollout_engines = [worker for worker in self.rollout_workers if ray.get(worker.is_engine.remote())]
        self.engine_iter = itertools.cycle(iter(self.rollout_engines))

    def _forward_step(self, data: List[Dict], iteration, is_eval):
        # random.shuffle(data)
        MAX_CONCURRENT = int(0.75*len(data))
        # MAX_CONCURRENT = int(1*len(data))
        
        data_iter = iter(data)
        ref_to_info = {}
        for i in range(min(len(data), MAX_CONCURRENT)):
            data_item = next(data_iter)
            selected_engine = next(self.engine_iter)
            ref = selected_engine.generate.remote(**data_item, is_eval=is_eval)
            ref_to_info[ref] = {
                "engine": selected_engine,
                "data_item": data_item
            }
        
        output_list = []
        data_list = []

        while ref_to_info:
            ready_refs, _ = ray.wait(list(ref_to_info.keys()), num_returns=1)
            ready_ref = ready_refs[0]
            output = ray.get(ready_ref)
            info = ref_to_info.pop(ready_ref)
            engine = info["engine"]
            data_item = info["data_item"]
            output_list.append(output)
            data_list.append(data_item)
            try:
                next_data_item = next(data_iter)
                new_ref = engine.generate.remote(**next_data_item, is_eval=is_eval)
                ref_to_info[new_ref] = {
                    "engine": engine,
                    "data_item": next_data_item
                }
            except StopIteration:
                pass
        
        results = ray.get(self.rollout_engines[0].postprocess_func.remote(output_list, data_list))
        trainer_dp_size = self.global_args.models.policy_trainer.replica_dp_size * self.global_args.models.policy_trainer.num_replica
        batch_size_per_dp = len(data) / trainer_dp_size
        results = data_balance(results, trainer_dp_size, batch_size_per_dp)
        return results


    @compute_decorator(trainable=False, rollout=True)
    def forward_step(self, data: List[Dict], iteration=0, **kwargs):
        rets = self._forward_step(data, iteration, False)
        return rets

    @compute_decorator(trainable=False, rollout=True)
    def eval_forward(self, data: List[Dict], iteration=0, **kwargs):
        rets = self._forward_step(data, iteration, False)
        return rets

    def onload(self):
        ray.get([engine.onload.remote() for engine in self.rollout_engines])

    def offload(self):
        ray.get([engine.offload.remote() for engine in self.rollout_engines])
