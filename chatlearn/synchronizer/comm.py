# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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
"""Sync parameters"""
from typing import List, Tuple, TYPE_CHECKING, Optional

import torch
from torch import distributed as dist
from torch.multiprocessing.reductions import reduce_tensor

from chatlearn.utils import future
from chatlearn.utils.logger import logger
from .structs import (
    SynchronizerType,
    BucketInfo,
    SyncIteration
)


if TYPE_CHECKING:
    from chatlearn.models.base_module import BaseModule

class GeneralCommunicator:
    """
        Execute parameter synchronization based on the given sync plan. 
        For simplicity, the communicator claims that each actor 
        should either be one of sender, forwarder (not implemented yet) or
        receiver, where forwarder is extra actors created with a global PG
        for parameter synchronization if sender/receiver does not have one.

        Currently, in the synchronization process, given a sync plan generated
        by the planner:
        (1) sender will prepare the corresponding weights according to the 
        send_layout of each bucket.
        (2) if senders have a global PG, do an all-to-all on all-senders to
        exchange the buckets by colocated receivers.
        (3) send buckets to colocated receivers by IPC Handle.
        (4) if senders do not have PG and receivers have a global PG, do an 
        all-to-all on all-receivers to collect the required buckets.
        (5) call module.update_weights_from_buckets() to update the weights.

        In each iteration, the parameter synchronization applys the following
        operation for each parameter logically in the bucket:
        >>>    dst_shard_info.index(dst_weight).copy_(
        >>>         src_shard_info.index(src_weight)
        >>>    )
    """
    def __init__(
        self,
        model: 'BaseModule',
        local_plan: List[SyncIteration],
        synchronizer_type: SynchronizerType,
        in_process_group: bool = False,
        *,
        colocate_handle = None
    ):
        """The general implementation of the communicator. A communicator is reponsible
        for send/recv of a specific set of parameters based on the local plan it is
        initialized with.

        Args:
            model (BaseModule): The module this communicator attached to.
            local_plan (List[SyncIteration]): The plan made by some planner.
            synchronizer_type (SynchronizerType): The type of synchronizer. 
            See `SynchronizerType`
            in_process_group (bool, optional): Whether this communicator is in the PG.
            Defaults to False.
            colocate_handle (optional): A Ray handle of the colocated model if possible. 
            Defaults to None.
        """
        self.model = model
        self.plan: List[SyncIteration] = local_plan
        self.type = synchronizer_type

        # NOTE: rank represents the rank of the colocated actor in the PG
        self.rank, self.world_size = None, None
        if in_process_group:
            self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.colocate_handle = colocate_handle

    def prepare_send_buckets(
        self,
        iter_idx: int
    ) -> Tuple[List[Optional[BucketInfo]], Optional[BucketInfo]]:
        """Prepare buckets for sending data to other actors of 
        iteration `iter_idx`.

        Args:
            iter_idx (int): The iteration index.
            world_size (int): The world size of ProcessGroup.

        Returns:
            List[Optional[BucketInfo]]: a list of BucketInfo. The k-th element 
            is None if this rank does not send any data to rank k.
        """
        send_buckets = [None] * self.world_size
        this_rank_bucket = None
        if self.type != SynchronizerType.SEND:
            raise NotImplementedError()
        # NOTE: sender prepare local send buckets
        send_plan = self.plan[iter_idx].send_buckets
        for bucket_info, ranks in send_plan.items():
            assert bucket_info.buffer is None, "Double allocation of send buffer."
            bucket_info = bucket_info.copy()
            bucket_info.buffer = torch.empty(
                bucket_info.size,
                dtype=torch.uint8,
                device=torch.cuda.current_device(),
            )
            for offset, shard_info in bucket_info.send_layout:
                # NOTE: dtype of some weights may change during training, force data
                # dtype to be shard_info.dtype (the original dtype of the weight)
                # to avoid potential issue.
                shard = shard_info.index(
                    self.model.param_id_to_parameters[shard_info.param_id].to(shard_info.dtype)
                ).view(dtype=torch.uint8)
                bucket_info.buffer[offset:offset + shard_info.size].view(shard.shape).copy_(shard)
            for rank in ranks.values:
                if send_buckets[rank] is not None:
                    raise ValueError(f"Rank {rank} has multiple buckets")
                send_buckets[rank] = bucket_info
        this_rank_bucket = send_buckets[self.rank]
        send_buckets[self.rank] = None
        return send_buckets, this_rank_bucket

    def prepare_recv_buckets(
        self,
        iter_idx: int,
        world_size: int,
        alloc_buffer: bool=True
    ) -> List[Optional[BucketInfo]]:
        """Prepare buckets for receiving data from other actors of 
        iteration `iter_idx`.

        Args:
            iter_idx (int): The iteration index.
            world_size (int): The world size of ProcessGroup.
            alloc_buffer (bool, optional): Whether to create a contiguous 
            buffer to receive data. Defaults to True.

        Returns:
            List[Optional[BucketInfo]]: a list of BucketInfo. The k-th element 
            is None if this rank does not receive any data from rank k.
        """
        recv_plan = self.plan[iter_idx].recv_buckets
        recv_buckets = [None] * world_size
        for recv_bucket, src_rank in recv_plan.items():
            if src_rank == self.rank:
                # NOTE: noneed to prepare recv bucket from colocated sender
                continue
            if recv_bucket.buffer is not None:
                raise ValueError("Expect buffer of recv bucket to be None")
            recv_buckets[src_rank] = recv_bucket.copy()
            if alloc_buffer:
                recv_buckets[src_rank].buffer = torch.empty(
                    recv_buckets[src_rank].size,
                    dtype=torch.uint8,
                    device=torch.cuda.current_device(),
                )
        return recv_buckets

    @torch.no_grad()
    def all2all_sync_step(self, iter_idx):
        """Core communication for parameter synchronization."""
        # NOTE: if a receiver is not in PG, it should get handles by calling
        # `all2all_sync_step` of colocated model
        if self.type == SynchronizerType.RECV and self.rank is None:
            handles = future.wait(
                self.colocate_handle.remote('all2all_sync_step', iter_idx=iter_idx),
                return_output=True
            )
            recv_buckets = self.prepare_recv_buckets(iter_idx, alloc_buffer=False, world_size=len(handles))
            # rebuild bucket from IPC handles
            for src_rank, handle in enumerate(handles):
                if handle is None:
                    continue
                rebuild_func, rebuild_args = handle
                recv_buckets[src_rank].buffer = rebuild_func(*rebuild_args)
            return recv_buckets

        # NOTE: otherwise, do all2all on the current actor
        send_buckets, this_rank_bucket = self.prepare_send_buckets(iter_idx)
        recv_buckets = self.prepare_recv_buckets(iter_idx, world_size=self.world_size)
        ops = self._build_p2p_ops(send_buckets, recv_buckets)
        torch.cuda.synchronize()
        if len(ops) > 0:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
            torch.cuda.synchronize()
        dist.barrier()
        recv_buckets[self.rank] = this_rank_bucket
        if self.type == SynchronizerType.RECV:
            # receiver only need to update on the local actor, thus
            # we do need convert to recv buckets to handles.
            return recv_buckets

        handles = [None] * len(recv_buckets)
        for i, bucket in enumerate(recv_buckets):
            if bucket is not None:
                handles[i] = reduce_tensor(bucket.buffer)
        return handles

    @torch.no_grad()
    def parameter_sync(self):
        """Perform parameter synchronization on each actor. 
        
        We define all actors into 3 categories: sender, 
        receiver and forwarder (not implemented yet):
            Sender: actors own source parameters.
            Receiver: actors require source parameters for updating.
            Forwarder: In some cases, the global PG does not exist on 
            sender/receiver, side forwarder actors are created on 
            every GPU with a global PG for communication.

        In case w/o forwarder, in the `parameter_sync`, receiver
        will iterate through the plan in `all2all_sync_step` and fetch
        parameter buckets from sender, then update its parameters with
        the payloads in the buckets.
        """
        if self.type == SynchronizerType.FORWARD:
            raise NotImplementedError("Forwarder is not supported yet")

        if self.type != SynchronizerType.RECV:
            return

        for iter_idx in range(len(self.plan)):
            recv_buckets = self.all2all_sync_step(iter_idx)
            self.model.update_weights_from_buckets(recv_buckets)
            for bucket in recv_buckets:
                if bucket is not None:
                    bucket.buffer = None
            self.release_ipc_resources()

    @torch.no_grad()
    def release_ipc_resources(self):
        """Release the IPC handles in the reverse order"""
        if self.colocate_handle:
            future.wait(
                self.colocate_handle.remote('release_ipc_resources'),
                return_output=True
            )
        else:
            torch.cuda.ipc_collect()

    def _build_p2p_ops(
        self,
        send_buckets: List[Optional[BucketInfo]],
        recv_buckets: List[Optional[BucketInfo]]
    ):
        """build p2p ops for batch_isend_irecv"""
        send_ops = []
        recv_ops = []
        for recv_rank, send_bucket in enumerate(send_buckets):
            if send_bucket is not None:
                send_ops.append(
                    dist.P2POp(
                        dist.isend,
                        tensor=send_bucket.buffer,
                        peer=recv_rank
                    )
                )
                logger.debug(f"RANK {dist.get_rank()} -> RANK {recv_rank}: send {send_bucket.size} bytes")
        for send_rank, recv_bucket in enumerate(recv_buckets):
            if recv_bucket is not None:
                recv_ops.append(
                    dist.P2POp(
                        dist.irecv,
                        tensor=recv_bucket.buffer,
                        peer=send_rank
                    )
                )
                logger.debug(f"RANK {dist.get_rank()} <- RANK {send_rank}: recv {recv_bucket.size} bytes")
        return send_ops + recv_ops
