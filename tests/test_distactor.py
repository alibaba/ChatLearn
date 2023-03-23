import torch
import rlhf
from rlhf.engine import Engine
from rlhf.model_wrapper import RLHFTorchModule
import time
import rlhf

rlhf.init()
class PolicyModel(RLHFTorchModule):

    def setup(self):
        time.sleep(0.05)

    def forward_step(self, data):
        #assert data['a'].device.type == 'cpu', data['a'].device.type
        time.sleep(0.1)
        return data

model = PolicyModel('policy')

engine = Engine(model)
engine.setup()
a = torch.ones([1])
b = torch.ones([1])
model = engine.models[0].replicas[0]
res0 = model.forward_step({'a': a, 'b': b})
res0 = rlhf.get(res0)[0]
res0 = model.forward_step({'a': a, 'b': b})
res0 = rlhf.get(res0)[0]
assert res0['a'].device.type == 'cpu', res0['a'].device

engine.logging_summary()

print(res0)




