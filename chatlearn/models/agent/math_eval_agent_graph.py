import copy
import json
from typing import (
    Annotated,
    Sequence,
    TypedDict,
    Any
)
import asyncio

from transformers import AutoTokenizer
from omegaconf import DictConfig
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool

from chatlearn.models.agent.agent_module import register
from chatlearn.models.agent.chat_model import CustomChatModel
from chatlearn.models.agent.base_agent_graph import BaseAgentGraph, AgentGraphOutput
from chatlearn.utils.rule_reward_score.math import last_boxed_only_string, remove_boxed, is_equiv

class AgentState(TypedDict):
    """The state of the agent."""
    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]

@register("matheval_agent")
class MathEvalAgentGraph(BaseAgentGraph):

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

        self.chatmodel = CustomChatModel(model=self.agent_name, llm=self.llm, tokenizer=self.tokenizer)
        
        # define node function
        async def call_model(
            state: AgentState,
            config: RunnableConfig,
        ):
            chatmodel = config['configurable']['chat_model']
            sampling_params = config['configurable']['sampling_params']
            message = await chatmodel.ainvoke(state["messages"], sampling_params=sampling_params)
            return {"messages": [message]}

        @tool
        def mathlighteval_reward(answer: str, **kwargs):
            """A tool for calculating the reward of mathlighteval."""
            if is_equiv(answer, kwargs['ground_truth']):
                return "The answer is correct."
            else:
                return "The answer is wrong."

        async def tool_node(state: AgentState, config: RunnableConfig):

            outputs = []
            
            for tool_call in state["messages"][-1].tool_calls:
                args = copy.deepcopy(tool_call["args"])
                args['ground_truth'] = config['configurable']['ground_truth']
                loop = asyncio.get_running_loop()
                try:
                    tool_result = await loop.run_in_executor(
                        None,
                        lambda: tools_by_name[tool_call["name"]].func(**args),
                    )
                    # tool_result = tools_by_name[tool_call["name"]].func(**args)
                    outputs.append(
                        ToolMessage(
                            content = tool_result,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )
                except:
                    outputs.append(
                        ToolMessage(
                            content = "tool execute error",
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
            if not last_message.tool_calls or ai_messages_cnt >= self.cfg.max_ai_message_turn or total_token_cnt >= self.cfg.max_total_token_length:
                return "end"
            # Otherwise if there is, we continue
            else:
                return "continue"
        
        # Define graph
        workflow = StateGraph(AgentState)

        # bind tool
        tools = [mathlighteval_reward]
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

    async def run(self, messages, sampling_params: dict[str, Any], **kwargs) -> AgentGraphOutput:
        
        config = {
            "configurable": {
                "chat_model": self.chatmodel,
                "sampling_params": sampling_params,
                "ground_truth": kwargs['ground_truth']
            }
        }

        output = await self.graph.ainvoke(
            input = {'messages': messages},
            config=config
        )
        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(
            None,
            lambda: self.convert_agent_graph_output(output)
        )
        return output


if __name__ == "__main__":
    import sglang
    from typing import Annotated
    from typing_extensions import TypedDict
    import os
    import pprint
    import asyncio

    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables import RunnableConfig
    llm = sglang.Engine(model_path="Qwen3-8B")
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 2048}
    tokenizer = AutoTokenizer.from_pretrained(
            "Qwen3-8B", trust_remote_code=True
        )
    agentgraph = MathEvalAgentGraph(agent_name="debughh", cfg=None, llm=llm, tokenizer=tokenizer)
    system_prompt = (
                    "You are a math expert. You are given a question and you need to solve it step by step. "
                    "Reasoning step by step before any tool call. "
                    "You should use the `mathlighteval_reward` tool after step by step solving the question, "
                    "before generate final answer at least once and refine your answer if necessary. "
                    "Put your final answer within \\boxed{}."
                )
    messages = [{"role": "system", "content":system_prompt}, {"role": "user", "content": "At constant temperature, the pressure of a sample of gas is inversely proportional to its volume. I have some hydrogen in a 3.67 liter container with a pressure of 4 kPa. If I move all of it to a 1.835 liter container at the same temperature, what will the new pressure be in kPa? You must use the `mathlighteval_reward` tool after step by step solving the question"}]
    out = asyncio.run(agentgraph.run(messages=messages, sampling_params=sampling_params, ground_truth="8"))
    # messages = [{"role": "system", "content":system_prompt}, {"role": "user", "content": "What is the smallest value of $x$ such that $|5x - 1| = |3x + 2|$? Express your answer as a common fraction. You must use the `mathlighteval_reward` tool after step by step solving the question"}]
    # out = asyncio.run(agentgraph.run(messages=messages, sampling_params=sampling_params, ground_truth="-\\frac{1}{8}"))
    # messages = [{"role": "system", "content":system_prompt}, {"role": "user", "content": "What is the value of $\\frac{5!\\cdot2!}{3!}$ You must use the `mathlighteval_reward` tool after step by step solving the question"}]
    # out = asyncio.run(agentgraph.run(messages=messages, sampling_params=sampling_params, ground_truth="40"))
    print(out.str_output)

    # for item in out['messages']:
    #     print(item)
    # print(len(out['messages']))
    




