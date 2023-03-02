import rlhf
from rlhf.engine import Engine
from gpt_megatron import GPTMegatron

rlhf.init()
policy = GPTMegatron("ppo_policy")
engine = Engine(policy)

engine.setup()
res = engine.models[0].train()
rlhf.get(res)
