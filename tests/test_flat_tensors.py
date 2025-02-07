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

    def test_uncontiguouse_tensor(self):
        def _check_ta(t):
            self.assertEqual(t.reshape(-1).tolist(), list(range(12)))
            self.assertEqual(t.shape, torch.Size((3, 4)))
            self.assertEqual(t.stride(), torch.Size((1, 3)))

        def _check_tb(t):
            self.assertEqual(t.reshape(-1).tolist(), list(range(100, 112)))
            self.assertEqual(t.shape, torch.Size((3, 4)))
            self.assertEqual(t.stride(), torch.Size((4, 1)))

        # col-major tensor: ta
        ta = torch.arange(start=0, end=12, device='cuda').view(3, 4).t().contiguous().t()
        # row-major tensor: tb
        tb = torch.arange(start=100, end=112, device='cuda').view(3, 4)
        flatted = FlatTensors([ta, tb], primary_store_device='cpu')
        _check_ta(ta)
        _check_tb(tb)

        flatted.copy_to_primary_store()
        # ta is offloaded to CPU
        self.assertEqual(ta.numel(), 0)
        self.assertEqual(ta.device.type, 'cpu')

        # tb is offloaded to CPU
        self.assertEqual(tb.numel(), 0)
        self.assertEqual(tb.device.type, 'cpu')

        flatted.copy_to_gpu_buffer()
        _check_ta(ta)
        _check_tb(tb)

        del flatted

    def test_torch_flatten(self):
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        def _check_ta(t):
            self.assertEqual(t.reshape(-1).tolist(), list(range(12)))
            self.assertEqual(t.shape, torch.Size((3, 4)))
            self.assertEqual(t.stride(), torch.Size((1, 3)))

        def _check_tb(t):
            self.assertEqual(t.reshape(-1).tolist(), list(range(100, 112)))
            self.assertEqual(t.shape, torch.Size((3, 4)))
            self.assertEqual(t.stride(), torch.Size((4, 1)))

        # col-major tensor: ta
        ta = torch.arange(start=0, end=12, device='cuda').view(3, 4).t().contiguous().t()
        # row-major tensor: tb
        tb = torch.arange(start=100, end=112, device='cuda').view(3, 4)

        print(f"{ta.dtype=} {ta.shape=} {ta.stride()} {ta=}")
        print(f"{tb.dtype=} {tb.shape=} {tb.stride()} {tb=}")

        flat = _flatten_dense_tensors((ta, tb))

        print(f"{flat.dtype=} {flat.shape=} {flat.stride()} {flat=}")

        _check_ta(ta)
        _check_tb(tb)

        # col-major tensor: ta2
        ta2 = torch.zeros((3, 4), device='cuda').t().contiguous().t()
        # row-major tensor: tb2
        tb2 = torch.zeros((3, 4), device='cuda')

        print(f"{ta2.dtype=} {ta2.shape=} {ta2.stride()} {ta2=}")
        print(f"{tb2.dtype=} {tb2.shape=} {tb2.stride()} {tb2=}")

        ta3, tb3 = _unflatten_dense_tensors(flat, (ta2, tb2))

        print(f"{ta3.dtype=} {ta3.shape=} {ta3.stride()} {ta3=}")
        print(f"{tb3.dtype=} {tb3.shape=} {tb3.stride()} {tb3=}")



if __name__ == '__main__':
    unittest.main()
