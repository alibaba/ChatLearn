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

import torch
from transformers import AutoTokenizer

from chatlearn import BaseModule
from chatlearn.data.prompt_dataset import PromptPipeline
from chatlearn.runtime.decorator import monitor_error, compute_decorator, timeit


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
                "str_outputs": str_outputs,
            }
        )
        data_output.append(input_data)

    print("str_outputs", data_output[0]["str_outputs"])
    print("data_sources", data_output[0]["data_source"])
    print("ground_truth", data_output[0]["ground_truth"])
    return data_output

class AgentManager(BaseModule):
    """Agent Module"""

    def __init__(self, name: str, args=None, replica_id: int=0):
        """ChatLearn main agent entrypoint
        """
        super().__init__(name, args=args, replica_id=replica_id)
        assert self.total_gpu == 0, "AgentManager does not require GPU"
        self._num_gpu_per_replica = 0
        self._num_replica = self.module_args.num_cpu // self.module_args.cpu_per_process

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
        # set shard variable
        ray.get(self.shared_var.set.remote('finish_cnt', 0))
        # self.engine2request_map = defaultdict(list)
        # for engine in self.rollout_engine:
        #     self.engine2request_map[engine] = []

    def get_rollout(self, namespace="policy"):
        all_actors = ray.util.list_named_actors(all_namespaces=True)
        rollout_actors = [actor for actor in all_actors if actor['namespace'] == namespace]
        self.rollout_workers = [ray.get_actor(**item) for item in rollout_actors]
        # rollout engine is not setup need to load later
        self.rollout_engines = [worker for worker in self.rollout_workers if ray.get(worker.is_engine.remote())]
        self.engine_iter = itertools.cycle(iter(self.rollout_engines))

    # def select_engine(self, request_id):
        # RoundRobin

        # # find request id whether in engine
        # least_request_engine = None
        # least_request_cnt = None
        # for engine, request_list in self.engine2request_map:
        #     if request_id in request_list:
        #         return engine
        #     if not least_request_engine or (len(request_list) < least_request_cnt):
        #         least_request_engine = engine
        #         least_request_cnt = len(request_list)
        
        # self.engine2request_map[least_request_engine].append(request_id)
        # return least_request_engine
        

    # def _forward_step(self, data: List[Dict], iteration, is_eval):
    #     outputs = ray.get(self.rollout_engines[0].generate.remote(data, is_eval))

    #     if outputs is not None:
    #         rets = sglang_postprocess_func(self.tokenizer, outputs, data)
    #         return rets
        
    def _forward_step(self, data: List[Dict], iteration, is_eval):
        ref_list = []
        for data_item in data:
            selected_engine = next(self.engine_iter)
            ref = selected_engine.generate.remote(**data_item, is_eval=is_eval)
            ref_list.append(ref)
        
        outputs = ray.get(ref_list)
        if outputs is not None:
            rets = agent_postprocess_func(outputs, data)
            return rets

    @compute_decorator(trainable=False, rollout=True)
    def forward_step(self, data: List[Dict], iteration=0, **kwargs):
        rets = self._forward_step(data, iteration, False)
        return rets

    @compute_decorator(trainable=False, rollout=True)
    def eval_forward(self, data: List[Dict], iteration=0, **kwargs):
        rets = self._forward_step(data, iteration, False)
        return rets

    def onload(self):
        finish_cnt = ray.get(self.shared_var.get.remote('finish_cnt'))
        finish_cnt += 1
        if finish_cnt == self._num_replica:
            for engine in self.rollout_engines:
                ray.get(engine.onload.remote())
            ray.get(self.shared_var.set.remote('finish_cnt', 0))
        else:
            ray.get(self.shared_var.set.remote('finish_cnt', finish_cnt))

    def offload(self):
        finish_cnt = ray.get(self.shared_var.get.remote('finish_cnt'))
        finish_cnt += 1
        if finish_cnt == self._num_replica:
            for engine in self.rollout_engines:
                ray.get(engine.offload.remote())
            ray.get(self.shared_var.set.remote('finish_cnt', 0))
        else:
            ray.get(self.shared_var.set.remote('finish_cnt', finish_cnt))