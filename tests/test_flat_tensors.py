import random
import unittest

import torch

from chatlearn.utils.flat_tensors import FlatTensors, BucketizedFlatTensors


# pylint: disable=missing-class-docstring
class TestFlatTensors(unittest.TestCase):

    @staticmethod
    def almost_same_memory_usage(t1, t2, eps):
        return abs(t1 / t2 - 1) < eps

    def run_flat_tensors_test_with_constructor(self, constructor):
        seed = 0
        random.seed(seed)
        torch.manual_seed(seed)

        measure1 = torch.cuda.memory_allocated()
        # Randomly generate some tensors.
        n = 4
        n_dims = [random.randint(1, 4) for _ in range(n)]
        shapes = [
            [random.randint(0, 8) for _ in range(dim)]
            for dim in n_dims
        ]

        tensors = [
            torch.rand(size=shape, device='cuda')
            for shape in shapes
        ]
        measure2 = torch.cuda.memory_allocated()
        tensors_usage = measure2 - measure1

        # Clone tensors for comparison.
        cloned = [
            tensor.detach().clone() for tensor in tensors
        ]
        measure3 = torch.cuda.memory_allocated()
        cloned_usage = measure3 - measure2
        self.almost_same_memory_usage(cloned_usage, tensors_usage, 1e-3)

        # Check after creating FlatTensors
        flat_tensor = constructor(tensors)
        for t, t_copied in zip(tensors, cloned):
            assert torch.equal(t, t_copied)

        # Check after offloaded.
        flat_tensor.copy_to_primary_store()
        for t in tensors:
            assert t.shape == torch.Size([0])

        measure4 = torch.cuda.memory_allocated()
        offloaded_memory = measure3 - measure4
        self.almost_same_memory_usage(offloaded_memory, tensors_usage, 1e-3)

        # Check after onloaded.
        flat_tensor.copy_to_gpu_buffer()
        measure5 = torch.cuda.memory_allocated()
        onloaded = measure5 - measure4
        self.almost_same_memory_usage(onloaded, tensors_usage, 1e-3)

        for t, t_copied in zip(tensors, cloned):
            assert torch.equal(t, t_copied)

    def test_flat_tensors(self):
        self.run_flat_tensors_test_with_constructor(
            lambda tensors: FlatTensors(tensors, primary_store_device='cpu')
        )
        torch.cuda.synchronize()

    def test_bucketized_flat_tensors(self):
        self.run_flat_tensors_test_with_constructor(
            lambda tensors: BucketizedFlatTensors(
                tensors, primary_store_device='cpu', bucket_size_mb=16
            )
        )
        torch.cuda.synchronize()


if __name__ == '__main__':
    unittest.main()
