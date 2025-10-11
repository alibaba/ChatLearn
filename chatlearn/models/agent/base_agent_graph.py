# pylint: disable=unused-argument
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
"""base agent graph"""
from abc import abstractmethod
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph
from omegaconf import DictConfig
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoProcessor
from torch import Tensor

from chatlearn.models.agent.chat_model import (CustomChatModel,
                                               find_last_ai_index,
                                               find_first_ai_index)
from chatlearn.models.sglang_module import AsyncEngine
from chatlearn.models.patches.transformers.qwen2_5_vl_patch import get_rope_index


def find_first_zero_group_end(lst):
    for i, x in enumerate(lst):
        if x != 0:
            return i - 1
    return len(lst) - 1 if lst else -1


# modified from https://github.com/volcengine/verl/blob/main/verl/experimental/agent_loop/agent_loop.py#L121
class AgentGraphOutput(BaseModel):
    """AgentGraphOutput"""
    str_output: str
    """total rollout str"""
    prompt_ids: list[int]
    """Prompt token ids."""
    all_token_ids: list[int]
    """all token ids including prompt, LLM generated token, tool response token."""
    loss_mask: list[int]
    """loss mask, 1 for LLM generated token, 0 for tool response token, input prompt."""
    response_logprobs: Optional[list[float]] = None
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    # multimodel data
    pixel_values: Any = None
    image_grid_thw: Any = None
    rope_deltas: Any = None
    position_ids: Any = None
    """processed multimodel batchfeature"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class BaseAgentGraph:
    """
    AgentGraph used for process a prompt by a self-defined LangGraph graph
    """

    def __init__(
        self,
        agent_name: str,
        cfg: DictConfig,
        llm: AsyncEngine,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor = None,
        model_type: str = "llm",
        **kwargs
    ):

        self.agent_name = agent_name
        self.cfg = cfg
        self.llm = llm
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_type = model_type

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentGraphOutput:
        raise NotImplementedError

    def build_graph(self) -> StateGraph:
        self.chatmodel = CustomChatModel(
            model=self.agent_name,
            llm=self.llm,
            tokenizer=self.tokenizer,
            processor=self.processor,
            model_type=self.model_type
        )

    def convert_agent_graph_output(self, messages: Dict) -> AgentGraphOutput:
        messages = messages["messages"]
        last_ai_message_idx = find_last_ai_index(messages)
        first_ai_message_idx = find_first_ai_index(messages)
        # discard messages after last ai message
        all_token_ids = messages[last_ai_message_idx].response_metadata["token_ids"]
        loss_mask = messages[last_ai_message_idx].response_metadata["loss_mask"]
        prompt_end_idx = find_first_zero_group_end(loss_mask)
        prompt_ids = all_token_ids[: prompt_end_idx + 1]
        num_turns = last_ai_message_idx + 1
        str_output = self.tokenizer.decode(all_token_ids[prompt_end_idx + 1 :])

        pixel_values = None
        image_grid_thw = None
        rope_deltas = None
        position_ids = None
        multimodel_batchfeature = messages[first_ai_message_idx].response_metadata.get("multimodel_batchfeature", None)
        if multimodel_batchfeature:
            pixel_values = multimodel_batchfeature.get("pixel_values")
            image_grid_thw = multimodel_batchfeature.get("image_grid_thw")
            position_ids, rope_deltas = get_rope_index(
                self.processor,
                input_ids=multimodel_batchfeature.get("input_ids"),
                image_grid_thw=multimodel_batchfeature.get("image_grid_thw"),
                video_grid_thw=multimodel_batchfeature.get("video_grid_thw"),
                second_per_grid_ts=multimodel_batchfeature.get("second_per_grid_ts"),
                attention_mask=multimodel_batchfeature.get("attention_mask"),
            )
            position_ids = position_ids.squeeze().tolist()
            # rope_deltas = multimodel_batchfeature.get("rope_deltas")
            # position_ids = multimodel_batchfeature.get("position_ids").squeeze().tolist()
        return AgentGraphOutput(
            str_output=str_output,
            prompt_ids=prompt_ids,
            all_token_ids=all_token_ids,
            loss_mask=loss_mask,
            num_turns=num_turns,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            rope_deltas=rope_deltas,
            position_ids=position_ids
        )
