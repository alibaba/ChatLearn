
import ray
import ray.util.collective as collective
import torch

import chatlearn


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self):
        from chatlearn.launcher.initialize import patch_ray
        patch_ray()
        self.send1 = torch.ones((4,), dtype=torch.bfloat16, device="cuda")
        self.send1 = torch.nn.parameter.Parameter(self.send1)
        self.recv1 = torch.zeros((4,), dtype=torch.bfloat16, device="cuda")
        self.recv1 = torch.nn.parameter.Parameter(self.recv1)
        self.rank = -1

    def setup(self, world_size, rank):
        self.rank = rank
        collective.init_collective_group(world_size, rank, "nccl", "8")
        return True

    def compute(self, src, tgt):
        if self.rank == 0:
            collective.send(self.send1*2, tgt, "8")
        else:
            collective.recv(self.recv1, src, "8")
        return self.recv1
    
    def compute2(self):
        if self.rank == 0:
            collective.send_multigpu(self.send2 * 4, 1, 0, "8")
        else:
            collective.recv_multigpu(self.recv1, 0, 1, "8")
        return self.recv1

    def recv(self, src_rank, src_gpu):
        collective.recv_multigpu(self.recv1, src_rank, src_gpu, "8")


    def recv2(self, src_rank, src_gpu):
        collective.recv_multigpu(self.recv2, src_rank, src_gpu, "8")
        return self.recv2

    def destroy(self):
        collective.destroy_collective_group("8")


def test_send_recv():
    num_workers = 3
    workers = []
    init_rets = []
    w0 = Worker.remote()
    init_rets.append(w0.setup.remote(num_workers, 0))
    w1 = Worker.remote()
    init_rets.append(w1.setup.remote(num_workers, 1))
    w2 = Worker.remote()
    init_rets.append(w2.setup.remote(num_workers, 2))

    workers = [w0, w1, w2]
    a = ray.get(init_rets)
    print('================== init done', a, flush=True)
    results = [w0.compute.remote(0, 1), w1.compute.remote(0, 1)]
    print(ray.get(results))
    print('send from w0 to w2', flush=True)
    results = [w0.compute.remote(0, 2), w2.compute.remote(0, 2)]
    print(ray.get(results))

    ray.get([w.destroy.remote() for w in workers])

TEST_CASE = [test_send_recv]