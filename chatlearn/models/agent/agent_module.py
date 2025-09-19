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
from typing import Any, Dict, List

import ray
import torch
from omegaconf import OmegaConf
from ray import ObjectRef

from chatlearn.models.agent.base_agent_graph import BaseAgentGraph
from chatlearn.models.agent.chat_model import CustomChatModel
from chatlearn.models.sglang_module import AsyncSGLangModule

_graph_registry: Dict[str, BaseAgentGraph] = {}


def register(agent_name: str):

    def decorator(subclass: type[BaseAgentGraph]) -> type[BaseAgentGraph]:
        _graph_registry[agent_name] = subclass
        return subclass

    return decorator


class AgentModule(AsyncSGLangModule):
    """agent module"""
    def __init__(self, name: str, args=None, replica_id: int = 0):
        """The chatlearn wrapper for a langgraph+async sglang model."""
        super().__init__(name, args=args, replica_id=replica_id)

        self.agent_factory: Dict[str, BaseAgentGraph] = {}
        self.chat_model: CustomChatModel = None

    def build_agent_graph(self, agent_name: str, agent_cfg_path: str) -> BaseAgentGraph:

        cfg = OmegaConf.load(agent_cfg_path) if agent_cfg_path else None
        graph_instance = _graph_registry[agent_name](
            agent_name=agent_name, cfg=cfg, llm=self.llm, tokenizer=self.tokenizer
        )
        self.agent_factory[agent_name] = graph_instance
        return graph_instance

    async def generate_per_request(self, query: Dict, is_eval: bool):
        # make sure key:messages in query
        assert (
            "agent_name" in query and query["agent_name"]
        ), "make sure set agent_name in dataset"
        agent_name = query["agent_name"]
        agent_cfg_path = query["agent_cfg_path"]
        graph = (
            self.agent_factory[agent_name]
            if agent_name in self.agent_factory
            else self.build_agent_graph(agent_name, agent_cfg_path)
        )
        sampling_params = self._get_sampling_params(is_eval)
        sampling_params["max_new_tokens"] = self.module_args.max_response_tokens_length
        output = await graph.run(sampling_params=sampling_params, **query)
        return output

    def postprocess_func(
        self,
        batched_outputs: List[Dict[str, Any]],
        input_data_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        # item in batched_outputs maybe ray ObjectRef to avoid cpu overhead
        if isinstance(batched_outputs[0], ObjectRef):
            batched_outputs = ray.get(batched_outputs)
        data_output = []
        for output, input_data in zip(batched_outputs, input_data_list):
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

        print("str_outputs", data_output[0]["str_outputs"])
        print("data_sources", data_output[0]["data_source"])
        print("ground_truth", data_output[0]["ground_truth"])
        return data_output
