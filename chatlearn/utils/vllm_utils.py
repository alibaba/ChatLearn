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
"""vllm utils"""
import os

from argparse import Namespace
from datetime import timedelta

import torch
import torch.distributed
from vllm.distributed.parallel_state import init_world_group
from vllm.distributed import parallel_state as mpu
from vllm.distributed.parallel_state import initialize_model_parallel
from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion

def initialize_vllm(args_dict):
    # pylint: disable=useless-return
    # Parse arguments
    args = Namespace(**args_dict)
    if not hasattr(args, 'distributed_backend'):
        args.distributed_backend = 'nccl'
    if not hasattr(args, 'distributed_timeout_minutes'):
        args.distributed_timeout_minutes = 10
    else:
        args.distributed_timeout_minutes = int(args.distributed_timeout_minutes)
    args.rank = int(os.getenv('RANK', '0')) if not hasattr(args, 'rank') else int(args.rank)
    args.local_rank = int(os.getenv('LOCAL_RANK', '0')) if not hasattr(args, 'local_rank') else int(args.local_rank)
    args.world_size = int(os.getenv("WORLD_SIZE", '1')) if not hasattr(args, 'world_size') else int(args.world_size)

    if args.rank == 0:
        print("> setting random seeds to {} ...".format(args.seed))

    if torch.distributed.is_initialized():
        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        world_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match args.world_size "
                f"({torch_world_size} vs. {args.world_size}).")
        return

    device_count = torch.cuda.device_count()
    if args.rank == 0:
        print('> initializing torch distributed ...', flush=True)
    # Manually set the device ids.
    if device_count > 0:
        device = args.rank % device_count
        if args.local_rank is not None:
            assert args.local_rank == device, \
                'expected local-rank to be the same as rank % device-count.'
        else:
            args.local_rank = device
        torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        timeout=timedelta(minutes=args.distributed_timeout_minutes))

    if CURRENT_VLLM_VERSION == VLLMVersion.v_0_8_5:
        _WORLD = None
        if _WORLD is None:
            ranks = list(range(torch.distributed.get_world_size()))
            _WORLD = init_world_group(ranks, args.local_rank, args.distributed_backend)
        else:
            assert _WORLD.world_size == torch.distributed.get_world_size(), (
                "world group already initialized with a different world size")
        mpu._WORLD = _WORLD

    initialize_model_parallel(args.tensor_model_parallel_size,
                              args.pipeline_model_parallel_size)

    return args

def vllm_use_v1():
    if "VLLM_USE_V1" not in os.environ:
        os.environ["VLLM_USE_V1"] = "1"
    return bool(int(os.getenv("VLLM_USE_V1", "1")))
