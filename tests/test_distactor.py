import torch

import chatlearn
from chatlearn.utils import future
from chatlearn.runtime.engine import BaseEngine
from chatlearn import TorchModule


class PolicyModel(TorchModule):

    def forward_step(self, data, iteration=0):
        #assert data['a'].device.type == 'cpu', data['a'].device.type
        return data


chatlearn.init()
model = PolicyModel('policy')

engine = BaseEngine(model)
engine.setup()
a = torch.ones([1])
b = torch.ones([1])
model = engine.models[0].replicas[0]
res0 = model.forward_step({'a': a, 'b': b})
res0 = future.get(res0)[0]
res0 = model.forward_step({'a': a, 'b': b})
res0 = future.get(res0)[0]
assert res0['a'].device.type == 'cpu', res0['a'].device

visible_devices = model.get_visible_gpus()
visible_devices = future.get(visible_devices)
assert visible_devices == [[0]], visible_devices

engine.logging_summary()

print(res0)
