# pylint: disable=arguments-differ,cell-var-from-loop,unnecessary-lambda,bare-except,missing-module-docstring,missing-class-docstring,import-outside-toplevel,unused-argument
import asyncio
import copy
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from omegaconf import DictConfig
from transformers import AutoTokenizer

from chatlearn.models.agent.agent_module import register
from chatlearn.models.agent.base_agent_graph import (AgentGraphOutput,
                                                     BaseAgentGraph)
from chatlearn.models.agent.chat_model import CustomChatModel


class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]


@register("mathcode_agent")
class MathCodeAgentGraph(BaseAgentGraph):

    def __init__(
        self,
        agent_name: str,
        cfg: DictConfig,
        llm: Any,
        tokenizer: AutoTokenizer,
        **kwargs
    ):
        super().__init__(agent_name, cfg, llm, tokenizer, **kwargs)
        self.build_graph()

    def build_graph(self) -> StateGraph:
        # lazy import
        # pip install agentscope==1.0.4 && pip install wandb==0.19.3
        from agentscope.tool import execute_python_code
        self.chatmodel = CustomChatModel(
            model=self.agent_name, llm=self.llm, tokenizer=self.tokenizer
        )

        # define node function
        async def call_model(
            state: AgentState,
            config: RunnableConfig,
        ):
            chatmodel = config["configurable"]["chat_model"]
            sampling_params = config["configurable"]["sampling_params"]
            message = await chatmodel.ainvoke(
                state["messages"], sampling_params=sampling_params
            )
            return {"messages": [message]}

        @tool
        async def execute_python(code: str):
            """execute python code and return execute result
            Return the result in the following format:
            <returncode>0</returncode><stdout>the print result in code</stdout><stderr>error message</stderr>

            Args:
                code (str): python code string, must include print to get result
            """
            result = await execute_python_code(code, timeout=5)
            return result.content[0]['text']


        async def tool_node(state: AgentState, config: RunnableConfig):

            outputs = []

            for tool_call in state["messages"][-1].tool_calls:
                args = copy.deepcopy(tool_call["args"])
                try:
                    tool_result = await tools_by_name[tool_call["name"]].coroutine(**args)
                    outputs.append(
                        ToolMessage(
                            content=tool_result,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )
                except:
                    outputs.append(
                        ToolMessage(
                            content="tool execute error",
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )
            return {"messages": outputs}

        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            ai_messages_cnt = 0
            total_token_cnt = 0
            for message in messages:
                if message.type == "ai":
                    ai_messages_cnt += 1
                    total_token_cnt = len(message.response_metadata["loss_mask"])
            # If there is no function call, then we finish
            if (
                not last_message.tool_calls
                or ai_messages_cnt >= self.cfg.max_ai_message_turn
                or total_token_cnt >= self.cfg.max_total_token_length
            ):
                return "end"
            # Otherwise if there is, we continue
            else:
                return "continue"

        # Define graph
        workflow = StateGraph(AgentState)

        # bind tool
        tools = [execute_python]
        tools_by_name = {tool.name: tool for tool in tools}
        self.chatmodel = self.chatmodel.bind_tools(tools)

        # add node
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        # add edges
        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            should_continue,
            {
                # If `tools`, then we call the tool node.
                "continue": "tools",
                # Otherwise we finish.
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
        self.graph = workflow.compile()

    async def run(
        self, messages, sampling_params: dict[str, Any], **kwargs
    ) -> AgentGraphOutput:

        config = {
            "configurable": {
                "chat_model": self.chatmodel,
                "sampling_params": sampling_params,
                "ground_truth": kwargs["ground_truth"],
            }
        }

        output = await self.graph.ainvoke(input={"messages": messages}, config=config)
        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(
            None, lambda: self.convert_agent_graph_output(output)
        )
        return output
