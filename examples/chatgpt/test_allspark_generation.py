import torch
import rlhf
from gpt_allspark import GPTAllSpark
from rlhf.engine import Engine

rlhf.init()
policy = GPTAllSpark("policy")
engine = Engine(policy)
engine.setup()

input_ids = torch.Tensor([[3143, 10574]]).to(torch.int64)
in_mask = torch.ones([1, 2], dtype=torch.int64)
torch_input = {
    "input_ids": input_ids,
    "attention_mask": in_mask,
}
res = engine.models[0].forward_step(torch_input)
print(rlhf.get(res))

