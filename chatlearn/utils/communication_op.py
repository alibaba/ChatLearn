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
"""Implementation of difference communication op"""
import torch
import torch.distributed as dist

_SP_PARALLEL_GROUP = None

def set_sp_parallel_group(group):
    global _SP_PARALLEL_GROUP
    _SP_PARALLEL_GROUP = group

def get_sp_parallel_group():
    return _SP_PARALLEL_GROUP

class SP_All2All_Single(torch.autograd.Function):
    """
    Autograd function for all_to_all_single (all_to_all to single tensor) using dist.all_to_all_single
    """
    @staticmethod
    def forward(ctx, input_tensor:torch.Tensor, sp_group:dist.ProcessGroup, split_dim:int, gather_dim:int):
        # now one of split_dim and gather_dim must be 1 which is the seq_dim
        assert split_dim == 1 or gather_dim == 1
        ctx.sp_group = sp_group
        ctx.split_dim = split_dim
        ctx.gather_dim = gather_dim
        sp_size = dist.get_world_size(group=sp_group)
        batch_size, _, _ = input_tensor.shape

        # [bsz, seq_len // sp, hidden_size] => [bsz, seq_len, hidden_size // sp]
        # Split on hidden dim               # [bsz, seq_len // sp, hidden_size] * sp
        # merge batch and seq_len           # [bsz * seq_len // sp, hidden_size // sp] * sp
        # concat on (b s) dim               # [bsz * seq_len, hidden_size // sp]
        # all_to_all                        # [bsz * seq_len, hidden_size // sp]
        # split on (b s) dim                # [bsz * seq_len, hidden_size // sp] * sp
        # reshape and concat on gather dim  # [bsz, seq_len, hidden_size // sp]
        # Reverse is the same, just change gather_dim and split_dim
        input_hp = torch.tensor_split(input_tensor, sp_size, dim=split_dim)
        input_hp = [rearrange(input_hp_single, 'b s h -> (b s) h') for input_hp_single in input_hp]
        input_hp = torch.concat(list(input_hp), dim=0)
        output_tensor = torch.empty_like(input_hp, dtype=input_tensor.dtype, device=input_tensor.device)
        dist.all_to_all_single(output=output_tensor, input=input_hp, group=sp_group, async_op=False)
        output_tensor = torch.tensor_split(output_tensor, sp_size, dim=0)
        output_tensor = [rearrange(output_tensor[i], '(b s) h -> b s h', b=batch_size) for i in range(sp_size)]
        output_tensor = torch.cat(output_tensor, dim=gather_dim)

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        #backward is exactly the opposite of forward
        return (SP_All2All_Single.apply(ctx.sp_group, grad_output, ctx.gather_dim, ctx.split_dim), None, None, None)

def all_to_all_single(input_tensor: torch.Tensor, sp_group: dist.ProcessGroup, split_dim: int, gather_dim: int):
    return SP_All2All_Single.apply(input_tensor, sp_group, gather_dim, split_dim)

class SP_All2All(torch.autograd.Function):
    """
    Autograd function for all_to_all using dist.all_to_all
    """
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, sp_group:dist.ProcessGroup, split_dim:int, gather_dim:int):
        # now one of split_dim and gather_dim must be 1 which is the seq_dim
        ctx.sp_group = sp_group
        ctx.split_dim = split_dim
        ctx.gather_dim = gather_dim
        sp_size = dist.get_world_size(group=sp_group)
        # Split tensor on split_dim
        # All2All with sp_group
        # Concat tensor on gather_dim
        input_hp = list(torch.tensor_split(input_tensor, sp_size, dim=split_dim))
        input_hp = [input_single.contiguous() for input_single in input_hp]
        output_list = [torch.empty_like(input_single) for input_single in input_hp]
        dist.all_to_all(output_tensor_list=output_list, input_tensor_list=input_hp, group=sp_group, async_op=False)
        output_tensor = torch.cat(output_list, dim=gather_dim)

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # Backward is exactly the opposite of forward, switch the input and output
        grad = SP_All2All.apply(grad_output, ctx.sp_group, ctx.gather_dim, ctx.split_dim)
        return (grad, None, None, None)

def all_to_all(input_tensor: torch.Tensor, sp_group: dist.ProcessGroup, split_dim: int, gather_dim: int):
    return SP_All2All.apply(input_tensor, sp_group, split_dim, gather_dim)

class SP_Gather(torch.autograd.Function):
    """
    Autograd function for gather using dist.all_gather
    """
    @staticmethod
    def forward(ctx, input_tensor, sp_group, gather_dim):
        ctx.sp_group = sp_group
        ctx.gather_dim = gather_dim
        sp_size = dist.get_world_size(sp_group)
        output_tensor = [torch.empty_like(input_tensor) for _ in range(sp_size)]
        dist.all_gather(tensor_list=output_tensor, tensor=input_tensor, group=sp_group)
        output_tensor = torch.concat(output_tensor, dim=gather_dim)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        sp_size = dist.get_world_size(ctx.sp_group)
        sp_local_rank = dist.get_rank(ctx.sp_group)
        grad = torch.tensor_split(grad_output, sp_size, dim=ctx.gather_dim)[sp_local_rank]
        return (grad, None, None)

def gather(input_tensor: torch.Tensor, sp_group: dist.ProcessGroup, gather_dim: int):
    return SP_Gather.apply(input_tensor, sp_group, gather_dim)
