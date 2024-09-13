import torch

import chatlearn
from chatlearn.utils import future
from chatlearn.runtime.engine import BaseEngine
from chatlearn import TorchModule


class PolicyModel(TorchModule):

    def forward_step(self, data, iteration):
        #assert data['a'].device.type == 'cpu', data['a'].device.type
        return data


class ReferenceModel(TorchModule):

    def forward_step(self, data, iteration):
        #assert data['a'].device.type == 'cpu', data['a'].device.type
        return data


chatlearn.init()
chatlearn.get_args().models["policy"].num_gpu = 4
chatlearn.get_args().models["policy"].tensor_model_parallel_size = 2
chatlearn.get_args().models["reference"].num_gpu = 4
chatlearn.get_args().models["reference"].tensor_model_parallel_size = 2
chatlearn.get_args().models["policy"].gpu_per_process = 1
chatlearn.get_args().models["reference"].gpu_per_process = 1
chatlearn.get_args().runtime_args.colocation = [["policy", "reference"]]

model = PolicyModel('policy')
model2 = ReferenceModel("reference")
engine = BaseEngine(model, model2)
engine.setup()
a = torch.ones([1])
b = torch.ones([1])
model = engine.models[0]
model2 = engine.models[1]

for replica_id in range(len(model.replicas)):
    visible_devices = future.get(model.replicas[replica_id].get_visible_gpus())
    if replica_id == 0:
        assert visible_devices == [[0], [1]], visible_devices
    else:
        assert visible_devices == [[2], [3]], visible_devices
    print(visible_devices)
    visible_devices = future.get(model2.replicas[replica_id].get_visible_gpus())
    if replica_id == 0:
        assert visible_devices == [[0], [1]], visible_devices
    else:
        assert visible_devices == [[2], [3]], visible_devices
    print(visible_devices)
