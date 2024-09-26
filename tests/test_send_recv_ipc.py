
import ray
import torch

# import chatlearn
from chatlearn.models.utils.communicator import Communicator
from chatlearn.data.storage import Storage
from chatlearn.utils import future

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, storage):
        self.comm = Communicator()
        self.rank = -1
        self._storage = storage
        
    def put(self, key, data):
        self._storage.put.remote(key, data)

    def get(self, key):
        ref = self._storage.get.remote(key)
        return future.get(ref)

    def send(self):
        tensor = torch.ones([1,2,3], dtype=torch.uint8, device="cuda") * 3
        handle = self.comm.ipc_send(tensor)
        group_name = "test"
        name = "tensor0"
        self.put(group_name + ":" + name, handle)
        return tensor
    
    def recv(self):
        handle_ref = None
        tensor = torch.zeros([1,2,3], dtype=torch.uint8, device="cuda")
        group_name = "test"
        name = "tensor0"
        while handle_ref is None:
            handle_ref = self.get(group_name + ":" + name)
        self.comm.ipc_recv(handle_ref, tensor)
        return tensor

if __name__ == "__main__":
    #chatlearn.init()
    num_workers = 3
    storage = Storage.remote()
    workers = []
    init_rets = []
    w0 = Worker.remote(storage)
    w1 = Worker.remote(storage)

    workers = [w0, w1]
    results = [w0.send.remote(), w1.recv.remote()]
    results = ray.get(results)
    print(results)
    assert (results[0] == results[1]).all()
    print('send from w0 to w1', flush=True)
