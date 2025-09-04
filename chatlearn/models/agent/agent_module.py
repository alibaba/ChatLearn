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

