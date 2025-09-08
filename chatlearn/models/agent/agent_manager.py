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
"""agent module"""
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

def group_dictionaries(data, num_groups=8, group_size=256, length_key='all_token_length'):
    """
    将字典列表按 length_key 的值进行负载均衡分组，每组最多 group_size 个元素。
    
    参数:
    - data: 字典列表，每个字典包含 length_key 字段
    - num_groups: 分成多少组（默认16）
    - group_size: 每组最多多少个元素（默认128）
    - length_key: 用于表示长度的键名（如 'length', 'len' 等）
    
    返回:
    - 列表，包含num_groups个组，每组是字典的列表
    """
    # 按 length_key 从大到小排序（优先处理长的）
    sorted_data = sorted(data, key=lambda x: x[length_key], reverse=True)
    
    # 初始化最小堆：(当前总长度, 组索引, 组内元素列表)
    heap = [(0, i, []) for i in range(num_groups)]
    heapq.heapify(heap)
    
    # 存储已满的组
    full_groups = []
    
    # 贪心分配
    for item in sorted_data:
        # 如果堆为空，说明所有组都满了
        if not heap:
            break
            
        # 取出当前总长度最小的组
        total_len, idx, group = heapq.heappop(heap)
        
        # 添加当前元素到组中
        group.append(item)
        new_total_len = total_len + item[length_key]
        
        # 检查该组是否已满
        if len(group) >= group_size:
            # 组已满，加入到full_groups，不再放回堆中
            full_groups.append((new_total_len, idx, group))
        else:
            # 组未满，放回堆中继续参与分配
            heapq.heappush(heap, (new_total_len, idx, group))
    
    # 合并结果：堆中未满的组 + 已满的组
    all_groups = []
    # 添加堆中剩余的未满组
    while heap:
        total_len, idx, group = heapq.heappop(heap)
        all_groups.append((idx, group))
    
    # 添加已满的组
    for total_len, idx, group in full_groups:
        all_groups.append((idx, group))
    
    # 按组索引排序并提取结果
    result = [group for idx, group in sorted(all_groups, key=lambda x: x[0])]

    flattened_result = [item for sublist in result for item in sublist]
    
    return flattened_result


def sglang_postprocess_func(
    tokenizer: AutoTokenizer,
    batched_outputs: List[Dict[str, Any]],
    input_data_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    data_output = []
    for output, input_data in zip(batched_outputs, input_data_list):
        prompt_token_ids = input_data["input_ids"]
        output_tokens = output["output_ids"]
        response_token_length = output["meta_info"]["completion_tokens"]
        prompt_token_length = output["meta_info"]["prompt_tokens"]
        str_outputs = tokenizer.decode(output_tokens, skip_special_tokens=True)
        all_tokens = torch.tensor(prompt_token_ids + output_tokens)
        input_data.update(
            {
                "prompt_token_ids": prompt_token_ids,
                "all_tokens": all_tokens,
                "response_token_length": response_token_length,
                "prompt_token_length": prompt_token_length,
                "str_outputs": str_outputs,
            }
        )
        data_output.append(input_data)

    print("str_outputs", data_output[0]["str_outputs"])
    print("data_sources", data_output[0]["data_source"])
    print("ground_truth", data_output[0]["ground_truth"])
    return data_output

def agent_postprocess_func(
    batched_outputs: List[Dict[str, Any]],
    input_data_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    data_output = []
    for output,input_data in zip(batched_outputs, input_data_list):
        prompt_token_ids = output.prompt_ids
        response_token_length = len(output.all_token_ids) - len(output.prompt_ids)
        prompt_token_length = len(output.prompt_ids)
        str_outputs = output.str_output
        all_tokens = torch.tensor(output.all_token_ids)
        loss_mask = torch.tensor(output.loss_mask)
        input_data.update(
            {
                "loss_mask": loss_mask,
                "prompt_token_ids": prompt_token_ids,
                "all_tokens": all_tokens,
                "response_token_length": response_token_length,
                "prompt_token_length": prompt_token_length,
                "all_token_length": response_token_length + prompt_token_length,
                "str_outputs": str_outputs,
            }
        )
        data_output.append(input_data)

    # print("str_outputs", data_output[0]["str_outputs"])
    # print("data_sources", data_output[0]["data_source"])
    # print("ground_truth", data_output[0]["ground_truth"])
    return data_output

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

    def build_dataset(self, prompts: List[Dict], is_eval=False):

        seq_length = self.global_args.models.policy.get("seq_length")
        prompts_dataset = PromptPipeline(
            prompts,
            seq_length,
            self.tokenizer,
            enable_thinking=self.global_args.models.policy.get("enable_thinking", False),
        )

        return prompts_dataset

    @monitor_error()
    def setup(self):
        self.stats = {}
        self._metric_prefix = "agent"
        self.get_rollout()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.global_args.models.policy.get("load"), trust_remote_code=True
        )


    def get_rollout(self, namespace="policy"):
        all_actors = ray.util.list_named_actors(all_namespaces=True)
        rollout_actors = [actor for actor in all_actors if actor['namespace'] == namespace]
        self.rollout_workers = [ray.get_actor(**item) for item in rollout_actors]
        # rollout engine is not setup need to load later
        self.rollout_engines = [worker for worker in self.rollout_workers if ray.get(worker.is_engine.remote())]
        self.engine_iter = itertools.cycle(iter(self.rollout_engines))

    # def _forward_step(self, data: List[Dict], iteration, is_eval):
    #     outputs = ray.get(self.rollout_engines[0].generate.remote(data, is_eval))

    #     if outputs is not None:
    #         rets = sglang_postprocess_func(self.tokenizer, outputs, data)
    #         return rets
        
    def _forward_step(self, data: List[Dict], iteration, is_eval):
        # random.shuffle(data)
        ref_list = []
        for data_item in data:
            selected_engine = next(self.engine_iter)
            ref = selected_engine.generate.remote(**data_item, is_eval=is_eval)
            ref_list.append(ref)
        
        outputs = ray.get(ref_list)
        # random.shuffle(outputs)
        # dist.barrier()
        if outputs is not None:
            rets = agent_postprocess_func(outputs, data)
            return rets


    # def _forward_step(self, data: List[Dict], iteration, is_eval):
    #     # random.shuffle(data)
    #     MAX_CONCURRENT = 1536
    #     data_iter = iter(data)
    #     ref_to_info = {}
    #     for i in range(min(len(data), MAX_CONCURRENT)):
    #         data_item = next(data_iter)
    #         selected_engine = next(self.engine_iter)
    #         ref = selected_engine.generate.remote(**data_item, is_eval=is_eval)
    #         ref_to_info[ref] = {
    #             "engine": selected_engine,
    #             "data_item": data_item
    #         }
    #     results = []

    #     while ref_to_info:
    #         ready_refs, _ = ray.wait(list(ref_to_info.keys()), num_returns=1)
    #         ready_ref = ready_refs[0]
    #         output = ray.get(ready_ref)
    #         info = ref_to_info.pop(ready_ref)
    #         engine = info["engine"]
    #         data_item = info["data_item"]
    #         results += agent_postprocess_func([output], [data_item])
    #         try:
    #             next_data_item = next(data_iter)
    #             new_ref = engine.generate.remote(**next_data_item, is_eval=is_eval)
    #             ref_to_info[new_ref] = {
    #                 "engine": engine,
    #                 "data_item": next_data_item
    #             }
    #         except StopIteration:
    #             # 所有请求已分配，不再提交新任务
    #             pass
    #     # random.shuffle(results)
    #     results = group_dictionaries(results)
    #     # dist.barrier()
    #     return results


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
