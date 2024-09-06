import torch

import chatlearn
from chatlearn.utils import future
from chatlearn.runtime.engine import BaseEngine
from chatlearn import TorchModule


class PolicyModel(TorchModule):

    def forward_step(self, data):
        #assert data['a'].device.type == 'cpu', data['a'].device.type
        return data


class ReferenceModel(TorchModule):

    def forward_step(self, data):
        #assert data['a'].device.type == 'cpu', data['a'].device.type
        return data


chatlearn.init()
chatlearn.get_args().models["policy"].num_gpu = 2
chatlearn.get_args().models["policy"].tensor_model_parallel_size = 2
chatlearn.get_args().models["reference"].num_gpu = 2
chatlearn.get_args().models["reference"].tensor_model_parallel_size = 2

model = PolicyModel('policy')
model2 = ReferenceModel("reference")
engine = BaseEngine(model, model2)
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
