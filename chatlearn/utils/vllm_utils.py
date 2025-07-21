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

import argparse
import os
import re

from datetime import timedelta

import torch
import torch.distributed
from vllm.distributed.parallel_state import init_world_group
from vllm.distributed import parallel_state as mpu
from vllm.distributed.parallel_state import initialize_model_parallel
from vllm.model_executor.model_loader.utils import get_model_architecture  as get_model_architecture_v2

from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion

def get_model_architecture(config):
    return get_model_architecture_v2(config)[0]

def get_pipeline_model_parallel_rank():
    return mpu.get_pp_group().rank_in_group

def get_pipeline_model_parallel_world_size():
    return mpu.get_pp_group().world_size

def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='vLLM Arguments',
                                     allow_abbrev=False)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Args from environment
    args.rank = int(os.getenv('RANK', '0'))
    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    return args


def _init_distributed_environment(args):
    """Initialize the distributed environment."""
    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
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

    if torch.distributed.is_initialized():
        world_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match args.world_size "
                f"({torch_world_size} vs. {args.world_size}).")
    else:
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


def initialize_vllm( # pylint: disable=dangerous-default-value,useless-return
    extra_args_provider=None,
    ignore_unknown_args=False,
    allow_no_cuda=False,
    args_dict=None
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."
    # Parse arguments
    args = parse_args(extra_args_provider, ignore_unknown_args)
    if args_dict:
        for key, value in args_dict.items():
            if value == "None":
                value = None
            if hasattr(args, key):
                default_value = getattr(args, key)
                if default_value is not None and value is not None:
                    default_type = type(default_value)
                    if not isinstance(value, default_type):
                        value = default_type(value)
            setattr(args, key, value)
    def finish_mpu_init():
        # Pytorch distributed.
        _init_distributed_environment(args)

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))


    finish_mpu_init()

    return args

def vllm_use_v1():
    if "VLLM_USE_V1" not in os.environ:
        os.environ["VLLM_USE_V1"] = "1"
    return bool(int(os.getenv("VLLM_USE_V1", "1")))
