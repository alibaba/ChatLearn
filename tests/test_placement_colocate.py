import time

import torch

import rlhf
from rlhf.utils import future
from rlhf import Engine
from rlhf import RLHFTorchModule

rlhf.init()
rlhf.get_args().models["policy"].num_device = 2
rlhf.get_args().models["reference"].num_device = 2
rlhf.get_args().models["policy"].gpu_per_process = 1
rlhf.get_args().models["reference"].gpu_per_process = 1
rlhf.get_args().rlhf_args.num_rollout_worker = 2
rlhf.get_args().rlhf_args.colocation = [["policy", "reference"]]

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

