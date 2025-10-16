# pylint: disable=unused-argument
"""base chat model warp for langgraph"""
import asyncio
import json
import uuid
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (AIMessage, BaseMessage,
                                     convert_to_openai_messages)
from langchain_core.messages.tool import InvalidToolCall, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field
from qwen_vl_utils import process_vision_info
from transformers.feature_extraction_utils import BatchFeature

from chatlearn.models.agent.tool_parser import ToolParser
from chatlearn.utils.logger import logger


def find_last_ai_index(messages):
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "ai":
            return i
    return -1

def find_first_ai_index(messages):
    for idx, message in enumerate(messages):
        if getattr(message, "type", None) == "ai":
            return idx
    return -1

class CustomChatModel(BaseChatModel):
    """CustomChatModel for async sglang"""
    model_name: str = Field(alias="model")
    llm: Any

    tokenizer: Any
    processor: Any
    model_type: str = "llm"
    # used for tool call
    max_parallel_calls: int = 1

    def bind_tools(self, tools, **kwargs) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the model.

        Args:
            tools: Sequence of tools to bind to the model.

        Returns:
            A Runnable that returns a message.
        """
        formatted_tools: list = [convert_to_openai_tool(tool) for tool in tools]
        for tool in formatted_tools:
            tool["function"]["parameters"]["properties"].pop("kwargs", None)

        # used to remove system prompt prefix when encoding tool response
        system_prompt = self.tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True
        )
        kwargs["system_prompt"] = system_prompt

        return self.bind(tools=formatted_tools, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate chat completion message.

        Args:
            messages (list[BaseMessage]): List of list of messages.
            stop (Optional[list[str]], optional): Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings. Defaults to None.

        Returns:
            ChatResult: Chat result.
        """
        processed_data = await self._preprocess(messages, **kwargs)
        image_data = processed_data.get("image_data", None)

        first_ai_message_idx = find_first_ai_index(messages)
        if first_ai_message_idx != -1:
            image_data = messages[first_ai_message_idx].response_metadata.get("image_data", None)

        assert "sampling_params" in kwargs, "please pass sampling_params(Dict)"
        sampling_params = kwargs["sampling_params"]
        output: Dict = await self.llm.async_generate(
            prompt=None,
            sampling_params=sampling_params,
            return_logprob=False,
            input_ids=processed_data.get("token_ids"),
            image_data = image_data
        )
        message = await self._postprocess(output, processed_data)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _preprocess(self, messages: list[BaseMessage], **kwargs: Any):
        """
        convert list[BaseMessage] to SGLangModule.generate input
        """
        # messages: [system], human, ai, human|tool, ai, human|tool, ...
        assert messages[-1].type in [
            "human",
            "tool",
        ], f"Last message must be human or tool, but got {messages[-1].type}"
        loop = asyncio.get_running_loop()

        # get image and video data
        # we assume multimodel data only appears in the first round
        # so we only process multimodel data in first round
        # tokenizer prompt after last AI message
        last_ai_message_idx = find_last_ai_index(messages)
        # not find ai message, means first input
        if last_ai_message_idx == -1:

            openai_messages = convert_to_openai_messages(messages)
            text = await loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    openai_messages,
                    tools=kwargs.get("tools"),
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=False,
                ),
            )
            if self.model_type == "llm":
                token_ids = await loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.encode(text)
                )
                return {"token_ids": token_ids, "loss_mask": [0] * len(token_ids)}
            elif self.model_type == "vlm":

                image_inputs, video_inputs = await loop.run_in_executor(
                    None,
                    lambda: process_vision_info(openai_messages)
                )
                # process multimodel data
                multimodel_batch_feature: BatchFeature = await loop.run_in_executor(
                    None,
                    lambda: self.processor(text=text, images=image_inputs, videos=video_inputs, return_tensors="pt")
                )
                token_ids = multimodel_batch_feature.get("input_ids")[0].tolist()
                loss_mask = [0] * len(token_ids)
                return {"token_ids": token_ids,
                        "loss_mask": [0] * len(token_ids),
                        "multimodel_batch_feature": multimodel_batch_feature,
                        "image_data": image_inputs,
                        "video_data": video_inputs
                        }
        # find ai message, only encode messages after last ai message
        else:
            remaining_messages = messages[last_ai_message_idx + 1 :]
            previous_token_ids = messages[last_ai_message_idx].response_metadata[
                "token_ids"
            ]
            previous_loss_mask = messages[last_ai_message_idx].response_metadata[
                "loss_mask"
            ]

            remaining_token_ids = await loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    convert_to_openai_messages(remaining_messages),
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                ),
            )
            remaining_token_ids = remaining_token_ids[len(kwargs["system_prompt"]) :]
            token_ids = previous_token_ids + remaining_token_ids
            loss_mask = previous_loss_mask + [0] * len(remaining_token_ids)
            return {"token_ids": token_ids, "loss_mask": loss_mask}

    async def _postprocess(self, output: Dict, processed_data: Dict):
        """convert sglang output to LangGraph AIMessage"""
        token_ids = processed_data["token_ids"]
        loss_mask = processed_data["loss_mask"]
        output_ids = output["output_ids"]
        completion_tokens = output["meta_info"]["completion_tokens"]
        # 198, 151667, 271, 151668, 271
        # !!!!!!!!!!!!! sglang 0.5.1.post2 output_ids has addintional \n<think>\n\n</think>\n\n
        if len(output_ids) != completion_tokens:
            output_ids = output_ids[-completion_tokens:]
        token_ids += output_ids
        loss_mask = loss_mask + [1] * len(output_ids)
        tool_parser = ToolParser.get_tool_parser("hermes", self.tokenizer)
        # TODO: remove duplicate operations
        content, function_calls = await tool_parser.extract_tool_calls(output_ids)

        tool_calls, invalid_tool_calls = [], []

        for function_call in function_calls:
            error = None
            try:
                args = json.loads(function_call.arguments)
                if not isinstance(args, dict):
                    error = f"Tool arguments must be a JSON object, got {type(args).__name__}"
            except json.JSONDecodeError as e:
                error = f"Invalid JSON tool arguments: {e}"

            if error:
                logger.warning(error)
                invalid_tool_calls.append(
                    InvalidToolCall(
                        name=function_call.name,
                        args=function_call.arguments,
                        id=str(uuid.uuid4()),
                        error=error,
                    )
                )
            else:
                tool_calls.append(
                    ToolCall(
                        name=function_call.name,
                        args=args,
                        id=str(uuid.uuid4()),
                    )
                )

        message = AIMessage(
            content=content,
            tool_calls=tool_calls[: self.max_parallel_calls],
            invalid_tool_calls=invalid_tool_calls[: self.max_parallel_calls],
            response_metadata={
                "token_ids": token_ids,
                "loss_mask": loss_mask,
                "multimodel_batch_feature": processed_data.get("multimodel_batch_feature", None),
                "image_data": processed_data.get("image_data", None),
                "video_data": processed_data.get("video_data", None)
            },
        )
        return message
