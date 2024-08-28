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
"""megatron utils"""

import functools
import re
import os
import subprocess

import torch

from chatlearn.utils.megatron_import_helper import get_args
from chatlearn.utils.megatron_import_helper import mpu
from chatlearn.utils.megatron_import_helper import parse_args, validate_args
from chatlearn.utils.megatron_import_helper import _load_base_checkpoint
from chatlearn.utils.megatron_import_helper import load_args_from_checkpoint
from chatlearn.utils.megatron_import_helper import load_checkpoint as megatron_load_checkpoint
from chatlearn.utils.megatron_import_helper import set_global_variables
from chatlearn.utils.megatron_import_helper import unwrap_model

from chatlearn.utils.megatron_import_helper import _initialize_distributed, _set_random_seed, _init_autoresume, _compile_dependencies

from chatlearn.utils.logger import logger


# regex to parse out layer number from param name
layer_re = re.compile(r'layers\.([0-9]+)')


def update_layer_num(layers_per_part, rank, m):
    # This assumes no interleaved pipeline execution
    layer = int(m.group(1))
    layer += rank * layers_per_part
    return f'layers.{layer}'


def build_pipeline_layer_name_mapping(src_layers_per_stage, src_rank, map_interval, tgt_last_stage, model, requires_grad):
    """
    remap pipeline layer_name. For each pipeline stage, the layer number starts with 0.
    Args:
        src_layers_per_stage: layer_per_stage in src model
        src_rank: src model pipeline rank
        map_interval: map interval from tgt to src, i.e. if src_layers_per_stage is 2, and tgt_layers_per_stage is 4,
                      then the map_iterval is tgt_layers_per_stage/src_layers_per_stage = 2
        tgt_last_stage: is target model in last stage
        model: megatron model
        requires_grad: whether the layer requires grad
    """
    name_mapping = {}
    for src_name, partition_param in model.named_parameters():
        if requires_grad:
            if not partition_param.requires_grad:
                continue
        if src_name.endswith("word_embeddings.weight") and "language_model" not in src_name:
            # See comment in MegatronModule.initialize_word_embeddings()
            if not tgt_last_stage:
                tgt_name = src_name.replace("word_embeddings.weight", "language_model.embedding.word_embeddings.weight")
            else:
                tgt_name = src_name
        else:
            # Translate destination layer number (0-N for each partition)
            # to source layer number (single-model layer number)
            rank = src_rank % map_interval
            _update_layer_num = functools.partial(update_layer_num, src_layers_per_stage, rank)
            tgt_name = re.sub(layer_re, _update_layer_num, src_name)
        name_mapping[tgt_name] = src_name
    return name_mapping

def initialize_megatron( # pylint: disable=dangerous-default-value
    extra_args_provider=None,
    args_defaults={},
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

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoints-args requires --load argument"
        load_args_from_checkpoint(args)

    validate_args(args, args_defaults)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

    args = get_args()
    if args.lazy_mpu_init:
        # TODO is this still a necessary option?
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        # No continuation function
        return None

def load_checkpoint(*_args, **kwargs):
    adaptive_parallel_strategy = False
    if "adaptive_parallel_strategy" in kwargs:
        adaptive_parallel_strategy = kwargs.pop("adaptive_parallel_strategy")
    if not adaptive_parallel_strategy:
        return megatron_load_checkpoint(*_args, **kwargs)
    args = get_args()
    target_tp = args.tensor_model_parallel_size
    target_pp = args.pipeline_model_parallel_size
    state_dict, _, _ = _load_base_checkpoint(args.load, rank0=True)
    args.iteration = state_dict['iteration']
    checkpoint_args = state_dict['args']
    checkpoint_tp = checkpoint_args.tensor_model_parallel_size
    checkpoint_pp = checkpoint_args.pipeline_model_parallel_size
    if target_tp != checkpoint_tp or target_pp != checkpoint_pp:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tools/megatron_checkpoint_utils.py")
        save_dir = args.load[:-1] if args.load.endswith("/") else args.load
        save_dir = save_dir + f"-transform-tp{target_tp}-pp{target_pp}"
        if not os.path.exists(save_dir):
            # use last rank so we can determin model_type by whether last pipeline stage contains pooler_head
            if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
                model_type = "GPT"
                for key in unwrap_model(_args[0])[0].state_dict().keys():
                    if 'pooler_head' in key:
                        model_type = "REWARD"
                        break
                cmd = f"python {script_path} --model-type {model_type} --load-dir {args.load} " + \
                      f"--save-dir {save_dir} --target-tensor-parallel-size {target_tp} --target-pipeline-parallel-size {target_pp}"
                logger.info(f"Transforming checkpoint for new parallel strategies {cmd}")
                subprocess.run(cmd, shell=True, check=True)
        torch.distributed.barrier()
        args.load = save_dir
        logger.info(f"Using transformed checkpoint {save_dir}")
    return megatron_load_checkpoint(*_args, **kwargs)
