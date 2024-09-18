# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""distributed utils"""

from collections import defaultdict
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def bucket_tensors(tensors, bucket_size_mb, buffer_num=None, tensor_changed=False):
    """Group tensors into chunks. We seperate sparse and dense tensor,
    each containing tensors of same type up to certain byte limit in total size.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.

    Return:
        dense_buckets: Blocks of tensors of same type and within size_limit.
        sparse_bucket: A list of sparse tensors
    """
    size_limit = bucket_size_mb * 1024 * 1024
    buf_dict = defaultdict(lambda: [[], 0])
    dense_buckets = []
    sparse_bucket = []
    for tensor in tensors:
        if tensor.is_sparse:
            sparse_bucket.append(tensor)
            continue
        t = tensor.type()
        size = tensor.numel() * tensor.element_size()
        buf_and_size = buf_dict[t]
        if size_limit > 0 and buf_and_size[1] + size > size_limit and buf_and_size[1] > 0: # pylint: disable=chained-comparison
            dense_buckets.append(buf_and_size[0])
            buf_and_size = buf_dict[t] = [[], 0]
        buf_and_size[0].append(tensor)
        buf_and_size[1] += size
    for buf, _ in buf_dict.values():
        if len(buf) > 0:
            dense_buckets.append(buf)
    return dense_buckets, sparse_bucket


def bucket_tensors_two_stage(tensors, bucket_size_mb, buffer_num=None, tensor_changed=False):
    """Group tensors into chunks. We seperate sparse and dense tensor,
    each containing tensors of same type up to certain byte limit in total size.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.

    Return:
        dense_buckets: Blocks of tensors of same type and within size_limit.
        sparse_bucket: A list of sparse tensors
    """
    size_limit = bucket_size_mb * 1024 * 1024
    buf_dict = defaultdict(lambda: [[], 0])
    dense_buckets = []
    sparse_bucket = []
    for idx, tensor in enumerate(tensors):
        buffer_multiple =  1 if buffer_num is None else buffer_num[idx]
        if tensor.is_sparse:
            sparse_bucket.append(tensor)
            continue
        t = tensor.type()
        # expand buffer size of dst ranks which recv tensor from trainer.
        size = tensor.numel() * tensor.element_size() * buffer_multiple
        buf_and_size = buf_dict[t]
        if size_limit > 0 and buf_and_size[1] + size > size_limit and buf_and_size[1] > 0: # pylint: disable=chained-comparison
            dense_buckets.append(buf_and_size[0])
            buf_and_size = buf_dict[t] = [[], 0]
        buf_and_size[0].append((torch.empty(size=[tensor.numel() * buffer_multiple],
                                      dtype=tensor.dtype,
                                      device=tensor.device) if (tensor_changed and buffer_multiple > 1) else tensor, [size // tensor.element_size(), buffer_multiple, tensor]))
        buf_and_size[1] += size
    for buf, size in buf_dict.values():
        if len(buf) > 0:
            dense_buckets.append(buf)
    return dense_buckets, sparse_bucket


def unflatten_dense_tensors(flat_tensors, tensors, sizes, num_ranks):
    all_buffers = defaultdict(list)

    offset = 0
    for size_multiple, tensor in zip(sizes, tensors):
        size, multiple, orig_tensor = size_multiple
        assert offset <= flat_tensors.numel()
        assert len(flat_tensors.shape) == 1
        flat_tensor = flat_tensors[offset:offset+size]
        per_size = size // multiple
        for rank in range(num_ranks):
            if multiple > 1:
                assert (flat_tensor.numel() //  multiple) == tensor.numel(), \
                    f"flat_tensor: {flat_tensor.shape} should be {multiple} times of tensor {orig_tensor.shape}, \
                        per_size: {per_size} total_size: {size} num_ranks: {num_ranks} offset: {offset}"
                all_buffers[rank].append(flat_tensor[rank * per_size:(rank + 1) * per_size].view(orig_tensor.shape))
            else:
                assert flat_tensor.numel() == orig_tensor.numel(), \
                    f"flat_tensor: {flat_tensor.shape} orig_tensor: {orig_tensor.shape}"
                all_buffers[rank].append(flat_tensor.view(orig_tensor.shape))
        del flat_tensor
        offset += size
    del flat_tensors
    return all_buffers



def coalesced_comm_dense(bucket, comm_call, extra_args, tensor_changed=True):
    """
    coalesced communication for dense parameters
    """
    flat_tensors = _flatten_dense_tensors(bucket)
    comm_call(flat_tensors, *extra_args)
    if tensor_changed:
        for tensor, synced in zip(
            bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def coalesced_comm_dense_two_stage(from_rank, to_rank, idx, bucket, comm_call, rank, extra_args, tensor_changed=True, stage2=False, index=0):
    """
    coalesced communication for dense parameters
    """
    all_tensors = []
    all_sizes = []
    num_ranks = 1
    orig_tensor_ele = 0
    orig_tensors = []
    for tensor, size in bucket:
        all_tensors.append(tensor)
        all_sizes.append(size)
        orig_tensors.append(size[2])
        orig_tensor_ele += size[2].numel()
        num_ranks = max(num_ranks, size[1])
    src_rank = extra_args[0]
    flat_tensors = _flatten_dense_tensors(all_tensors)
    comm_call(flat_tensors, *extra_args)
    if tensor_changed:
        index = 0 if stage2 else index
        all_buffers = unflatten_dense_tensors(flat_tensors, orig_tensors, all_sizes, num_ranks)
        for tensor, synced in zip(orig_tensors, all_buffers[index]):
            assert tensor.numel() == synced.numel(), f"rank {rank} tensor {tensor.shape} should be equal to synced.shape {synced.shape}, for all_sizes {all_sizes}"
            tensor.copy_(synced)
        del all_buffers[index]
        return all_buffers
    return None


def broadcast_var_object_dict(obj_dict, src_rank):
    if torch.distributed.get_rank() == src_rank:
        dict_as_list = list(obj_dict.items())
        list_length = len(dict_as_list)
        length_tensor = torch.tensor(list_length, device='cuda')
        torch.distributed.broadcast(length_tensor, src_rank)
        torch.distributed.broadcast_object_list(dict_as_list, src=src_rank)
        return obj_dict
    else:
        length_tensor = torch.tensor(0, device='cuda')
        torch.distributed.broadcast(length_tensor, src_rank)
        list_length = length_tensor.item()
        dict_as_list = [None] * list_length
        torch.distributed.broadcast_object_list(dict_as_list, src=src_rank)
        return dict(dict_as_list)
