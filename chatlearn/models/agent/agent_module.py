from typing import Optional, Dict, List, Any

import torch

from chatlearn.models.sglang_module import AsyncSGLangModule
from chatlearn.models.agent.math_eval_agent_graph import MathEvalAgentGraph


class AgentModule(AsyncSGLangModule):

    def __init__(self, name: str, args=None, replica_id: int = 0):
        """The chatlearn wrapper for a langgraph+async sglang model."""
        super().__init__(name, args=args, replica_id=replica_id)

        self.agent_factory = []
        self.chat_model = None


    def setup_engine(self):
        # setup sglang engine
        super().setup_engine()

        # construct chat model based on sglang AsyncEngine

        # to implement
        if self.is_engine():
            self.build_agent_graph("debugh")


    def build_agent_graph(self, agent_name: str):
        self.graph = MathEvalAgentGraph(agent_name=agent_name, llm=self.llm, tokenizer=self.tokenizer)

    async def generate(self, messages, is_eval: bool, **kwargs):
        sampling_params = self._get_sampling_params(is_eval)
        sampling_params["max_new_tokens"] = 2048
        output = await self.graph.run(messages=messages, sampling_params=sampling_params, gt=kwargs['ground_truth'])
        return output

    def postprocess_func(
        self,
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

        print("str_outputs", data_output[0]["str_outputs"])
        print("data_sources", data_output[0]["data_source"])
        print("ground_truth", data_output[0]["ground_truth"])
        return data_output
