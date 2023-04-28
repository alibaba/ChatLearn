import time

import torch

import rlhf
from rlhf import Engine
from rlhf import RLHFTorchModule

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

visible_devices = model.get_visible_gpus()
visible_devices = rlhf.get(visible_devices)
assert visible_devices == [[0]], visible_devices

engine.logging_summary()

print(res0)




