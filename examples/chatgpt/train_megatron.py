import rlhf
from rlhf import Engine
from gpt_megatron import GPTMegatron

rlhf.init()
policy = GPTMegatron("ppo_policy")
engine = Engine(policy)

engine.setup()
res = engine.models[0].replicas[0].train()
rlhf.get(res)
