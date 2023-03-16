import torch
import rlhf
from rlhf.engine import Engine
from rlhf.model_wrapper import RLHFModelWrapper

import rlhf

rlhf.init()
class PolicyModel(RLHFModelWrapper):
    
    def forward_step(self, data):
        assert data['a'].device.type == 'cuda', data['a'].device.type
        return data

model = PolicyModel('policy')

engine = Engine(model)

a = torch.ones([1])
b = torch.ones([1])

res0 = engine.models[0].replicas[0].forward_step({'a': a, 'b': b})
res0 = rlhf.get(res0)[0]
assert res0['a'].device.type == 'cpu', res0['a'].device




