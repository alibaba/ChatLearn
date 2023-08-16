import time

import torch

import chatlearn
from chatlearn.utils import future
from chatlearn import Engine
from chatlearn import RLHFTorchModule

chatlearn.init()
chatlearn.get_args().models["policy"].num_device = 2
chatlearn.get_args().models["policy"].tensor_model_parallel_size = 2
chatlearn.get_args().models["reference"].num_device = 2
chatlearn.get_args().models["reference"].tensor_model_parallel_size = 2

class PolicyModel(RLHFTorchModule):

    def setup(self):
        time.sleep(0.05)

    def forward_step(self, data):
        #assert data['a'].device.type == 'cpu', data['a'].device.type
        time.sleep(0.1)
        return data



class ReferenceModel(RLHFTorchModule):

    def setup(self):
        time.sleep(0.05)

    def forward_step(self, data):
        #assert data['a'].device.type == 'cpu', data['a'].device.type
        time.sleep(0.1)
        return data


model = PolicyModel('policy')
model2 = ReferenceModel("reference")
engine = Engine(model, model2)
engine.setup()
a = torch.ones([1])
b = torch.ones([1])
model = engine.models[0].replicas[0]
model2 = engine.models[1].replicas[0]
res0 = model.forward_step({'a': a, 'b': b})
res0 = future.get(res0)[0]
res0 = model.forward_step({'a': a, 'b': b})
res0 = future.get(res0)[0]
assert res0['a'].device.type == 'cpu', res0['a'].device

visible_devices = model.get_visible_gpus()
visible_devices = future.get(visible_devices)
visible_devices2 = model2.get_visible_gpus()
visible_devices2 = future.get(visible_devices2)
assert visible_devices == [[0], [1]], visible_devices
assert visible_devices2 == [[2], [3]], visible_devices2

engine.logging_summary()

print(res0)
