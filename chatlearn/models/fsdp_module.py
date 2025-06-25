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
"""FSDP  module"""
import os
import random
import functools
import gc
from contextlib import contextmanager, nullcontext

import ray
import numpy as np
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.distributed.fsdp import (MixedPrecision, ShardingStrategy, ShardedOptimStateDictConfig, ShardedStateDictConfig,
                                    FullStateDictConfig, StateDictType, FullyShardedDataParallel as FSDP)
from torch.distributed.fsdp.wrap import (size_based_auto_wrap_policy, transformer_auto_wrap_policy, lambda_auto_wrap_policy,
                                        _or_policy)
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.multiprocessing.reductions import reduce_tensor

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.trainer_pt_utils import get_module_class_from_name

from chatlearn.utils.logger import debug_rank_0
from chatlearn.utils.utils import dict_to_simplenamespace
from chatlearn.utils.communication_op import set_sp_parallel_group
from chatlearn.models.patches.monkey_patch import apply_sp_monkey_patch, apply_group_gemm
from .torch_module import TorchModule

from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

class FSDPModule(TorchModule):
    """TorchModule is the class for Alignment Torch models.

    Args
    ----
    name : str
        model name
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.fsdp_size = self.module_args.fsdp_size
        self.sp_size = self.module_args.ulysses_sequence_parallel_size
        self.device_mesh = None
        self.sp_device_mesh = None
        self.packing = self.module_args.packing
        self.max_token_in_seq = self.module_args.max_token_in_packing

    def get_visible_gpus(self):
        """
        :meta private:
        """
        return ray.get_gpu_ids()


    @staticmethod
    def init_fn(x: torch.nn.Module):
        if torch.distributed.get_rank() != 0:
            x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
            torch.cuda.empty_cache()
        return x

    @staticmethod
    def get_fsdp_wrap_policy(module:torch.nn.Module, min_num_params:int=0):
        """Get FSDP wrap policy for the module.

        Args:
            module: The module to get wrap policy for
            min_num_params: size based wrap policy min num params
        """

        default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
        fsdp_transformer_layer_cls_to_wrap = default_transformer_cls_names_to_wrap
        auto_wrap_policy = None

        policies = []

        # min_num_params must be 0 to use transformer_auto_wrap_policy
        if min_num_params > 0:
            size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
            policies.append(size_policy)
        elif fsdp_transformer_layer_cls_to_wrap is not None:
            transformer_cls_to_wrap = set()
            for layer_class in fsdp_transformer_layer_cls_to_wrap:
                transformer_cls = get_module_class_from_name(module, layer_class)
                assert transformer_cls is not None, "Could not find the transformer layer class to wrap in the model."
                transformer_cls_to_wrap.add(transformer_cls)

            transformer_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_cls_to_wrap,
            )
            policies.append(transformer_policy)
        ### hardcode for qwen2.5, fsdp warp for get submodule state dict ###
        def lambda_fn(sub_module: nn.Module):
            if sub_module in [module.model.embed_tokens, module.model.norm, module.lm_head]:
                return True
            return False
        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
        policies.append(lambda_policy)
        ## hardcode for qwen2.5, fsdp warp for get submodule state dict ##

        if len(policies) > 0:
            auto_wrap_policy = functools.partial(_or_policy, policies=policies)

        return auto_wrap_policy
    def create_device_mesh(self, world_size, fsdp_size):
        if not self.device_mesh:
            if world_size == fsdp_size:
                self.device_mesh = dist.device_mesh.init_device_mesh(
                    "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
                )
            else:
                self.device_mesh = dist.device_mesh.init_device_mesh(
                    "cuda",
                    mesh_shape=(fsdp_size, world_size // fsdp_size),
                    mesh_dim_names=["fsdp", "ddp"],
                )
            print(f"world size {world_size}, fsdp_size {fsdp_size}, {self.device_mesh}")

    def create_sp_device_mesh(self):
        # TODO: maybe this constrict can be eased out
        assert self.fsdp_size % self.sp_size == 0, \
            "fsdp_size must be divisible by sp_size"
        self.sp_device_mesh = dist.device_mesh.init_device_mesh(
            "cuda", mesh_shape=(self.world_size // self.sp_size, self.sp_size),
            mesh_dim_names=("dp", "sp")
        )
        set_sp_parallel_group(self.sp_device_mesh.get_group("sp"))

    def setup_distributed(self):
        print(self.get_dist_env())
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.create_device_mesh(self.world_size, self.fsdp_size)

    def peak_memory(self):
        """
        :meta private:
        """
        self._peak_memory = max(
            self._peak_memory, torch.cuda.max_memory_allocated() / (1024**3)
        )
        return self._peak_memory

    def empty_cache(self):
        """
        :meta private:
        """
        if not self.timers("empty_cache").started_:
            self.timers("empty_cache").start()
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        debug_rank_0(
            f"{self.name} replica: {self.replica_id}, before empty cache, peak mem: {peak_mem:.2f} GiB",
            self._logger,
        )
        # Manual gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        debug_rank_0(
            f"{self.name} replica: {self.replica_id}, after empty cache, peak mem: {peak_mem:.2f} GiB",
            self._logger,
        )
        self.timers("empty_cache").stop()

    def create_model(self, model_path: str , torch_dtype: torch.dtype, meta_init: bool) -> nn.Module:
        if not meta_init:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        else:
            model_config = AutoConfig.from_pretrained(model_path)
            with torch.device('meta'):
                model = AutoModelForCausalLM.from_config(
                    model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True
                )
        dist.barrier()
        return model

    @property
    def data_parallel_size(self):
        """
        :meta private:
        """
        if self.sp_device_mesh is not None:
            dp_group = self.sp_device_mesh.get_group('dp')
            return dist.get_world_size(group=dp_group)
        else:
            return dist.get_world_size()

    @property
    def data_parallel_rank(self):
        """
        :meta private:
        """
        if self.sp_device_mesh is not None:
            dp_group = self.sp_device_mesh.get_group('dp')
            return dist.get_rank(group=dp_group)
        else:
            return dist.get_rank()

    def tensor_parallel_rank(self):
        return self.data_parallel_rank

    def pipeline_parallel_rank(self):
        return 1

    def expert_model_parallel_size(self):
        return 1

    def check_sp_compatibility(self, config):
        assert config.num_attention_heads % self.sp_size == 0, \
            "num_attention_heads must be divisible by sp"
        if self.sp_size > config.num_key_value_heads:
            assert self.sp_size % config.num_key_value_heads == 0, \
                "When sp_size > num_key_value_heads, sp_size must be divisible by num_key_value_heads"

    def model_setup(self):
        """
        :meta private:
        """
        super().model_setup()
        self.setup_distributed()
        args = dict_to_simplenamespace(self.module_args)
        self.args = args

        local_rank = dist.get_rank()
        # When meta_init is enabled, we only load checkpoint on rank 0
        meta_init = self.module_args.meta_init and local_rank != 0
        model = self.create_model(args.load, torch_dtype=torch.bfloat16, meta_init=meta_init)
        if self.module_args.groupgemm:
            apply_group_gemm(model)
            dist.barrier()
        # Setup device mesh and apply patch for sequence parallel
        # Sequence_parallel should only be used during training
        if self.sp_size > 1:
            self.check_sp_compatibility(model.config)
            self.create_sp_device_mesh()
            apply_sp_monkey_patch(model.config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.load, trust_remote_code=True, use_fast=True
        )
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        # mix_precision_config = MixedPrecision(
        #     param_dtype=torch.bfloat16,
        #     reduce_dtype=torch.float32,
        #     buffer_dtype=torch.float32,
        # )
        mix_precision_config = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)
        # sharding_strategy = ShardingStrategy.FULL_SHARD
        # auto_wrap_policy = self.get_fsdp_wrap_policy(module = model)
        # If meta_init is True, other rank need to sync params from rank 0
        sync_module_states = self.module_args.meta_init
        # self.model = FSDP(
        #     model,
        #     cpu_offload=None,
        #     auto_wrap_policy=auto_wrap_policy,
        #     device_id=torch.cuda.current_device(),
        #     sharding_strategy=sharding_strategy,  # zero3
        #     mixed_precision=mix_precision_config,
        #     sync_module_states=sync_module_states,
        #     param_init_fn=FSDPModule.init_fn,
        #     device_mesh=self.device_mesh,
        #     forward_prefetch=False,
        # )
        fsdp_kwargs = {
            "mesh": self.device_mesh,
            "mp_policy": mix_precision_config,
            "reshard_after_forward": True,
        }
        default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
        fsdp_transformer_layer_cls_to_wrap = default_transformer_cls_names_to_wrap
        if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
            fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]
        modules = []
        for name, module in model.named_modules():
            if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings):
                modules.append(module)

        for idx, module in enumerate(modules):
            fully_shard(module, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module
        self.model = model

        self.model.to(torch.float32)
        FSDP.set_state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT)
        if not self.trainable:
            self.optimizer = None
            self.model.eval()
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),

                lr=self.module_args.optimizer.lr,
                betas=(self.module_args.optimizer.adam_beta1, self.module_args.optimizer.adam_beta2),
                weight_decay=self.module_args.optimizer.weight_decay
            )

        # resume model weights
        if self.resume_training:
            self.load_checkpoint(self._episode_id)

        self.offload()

    # def get_fsdp_param_name(self):
    #     name_list = []
    #     for name, _ in self.model.named_parameters():
    #         parts = name.split('.')
    #         filtered_parts = [
    #             part for part in parts
    #             if part not in {"_fsdp_wrapped_module", "_flat_param"}
    #         ]
    #         cleaned_name = '.'.join(filtered_parts)
    #         name_list.append(cleaned_name)
    #     return name_list

    # def get_fsdp_param_name(self):
    #     name_list = []
    #     for name, _ in self.model.named_parameters():
    #         name_list.append(name)
    #     return name_list

    # def get_weight_ipc_handles_by_name(self, block_name: str):
    #     """
    #     get fsdp warpped module weight by name get from named_parameters
    #     avoid get total model state_dict
    #     """
    #     torch.cuda.empty_cache()
    #     for prefix_name, module in self.model.named_modules():
    #         prefix_name = prefix_name.replace('_fsdp_wrapped_module.', '')
    #         if isinstance(module, FSDP) and prefix_name==block_name:
    #             state_dict = module.state_dict()
    #             reduce_tensor_dict = {}
    #             for name, param in state_dict.items():
    #                 reduce_tensor_dict['.'.join([prefix_name, name])] = reduce_tensor(param.full_tensor())
    #             return reduce_tensor_dict

    def get_fsdp_param_name(self):
        name_list = []
        for name, _ in self.model.named_parameters():
            name_list.append(name)
        return name_list

    def get_weight_ipc_handles_by_name(self, block_name: str):
        """
        get fsdp warpped module weight by name get from named_parameters
        avoid get total model state_dict
        """
        torch.cuda.empty_cache()
        for name, param in self.model.named_parameters():
            if name==block_name:
                reduce_tensor_dict = {}

                reduce_tensor_dict[name] = reduce_tensor(param.full_tensor().detach())
                return reduce_tensor_dict
    @torch.no_grad()
    def onload_weights(self, empty_cache=True):
        # _lazy_init(self.model, self.model)
        # assert self.model._is_root
        # device_id = torch.cuda.current_device()
        # for handle in self.model._all_handles:
        #     if handle._offload_params:
        #         continue
        #     flat_param = handle.flat_param
        #     handle.flat_param_to(torch.device(f"cuda:{device_id}"), non_blocking=True)
        #     # the following still keeps id(._local_shard) != id(.data)
        #     flat_param._local_shard = flat_param.data
        # if empty_cache:
        #     torch.cuda.empty_cache()
        device_id = torch.cuda.current_device()
        for param in self.model.parameters():
            param.data = param.data.to(torch.device(f"cuda:{device_id}"), non_blocking=True)

    @torch.no_grad()
    def offload_weights(self, empty_cache=True):
        # assert isinstance(self.model, FSDP)
        # # lazy init FSDP model
        # _lazy_init(self.model, self.model)
        # assert self.model._is_root, "Only support root model offloading to CPU"
        # for handle in self.model._all_handles:
        #     if handle._offload_params:
        #         continue
        #     flat_param = handle.flat_param
        #     assert (
        #         flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
        #         and id(flat_param.data) != id(flat_param._local_shard)
        #         and flat_param.data.size() == flat_param._local_shard.size()
        #     )
        #     handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        #     # Explicit call to free unshard flat param
        #     handle._free_unsharded_flat_param()
        #     # the following still keeps id(._local_shard) != id(.data)
        #     flat_param._local_shard = flat_param.data
        #     assert id(flat_param._local_shard) != id(flat_param.data)

        # # Explicit releas ipc handles
        # torch.cuda.ipc_collect()
        # if empty_cache:
        #     torch.cuda.empty_cache()
        for param in self.model.parameters():
            param.data = param.data.to(torch.device("cpu"), non_blocking=True)
        torch.cuda.ipc_collect()
        if empty_cache:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def offload_optimizer_states(self, empty_cache=True):
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)

        if empty_cache:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def onload_optimizer_states(self, empty_cache=True):
        if not self.optimizer.state:
            return
        device = torch.cuda.current_device()
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device, non_blocking=True)

        if empty_cache:
            torch.cuda.empty_cache()

    def save_checkpoint(self, iteration):
        save_dir = f"{self.runtime_args.output_dir}/save_model/{self.name}/{iteration}"
        if dist.get_rank() == 0 and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Make sure directory exists before writing
        dist.barrier()

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        # with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
        with nullcontext():
            model_state_dict = self.model.state_dict()
            optimizer_state_dict = self.optimizer.state_dict() if self.optimizer is not None else None
            # lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            extra_state_dict = {
                # "lr_scheduler": lr_scheduler_state_dict,
                "rng": self.get_rng_state(),
            }
            model_path = os.path.join(save_dir, f"model_world_size_{dist.get_world_size()}_rank_{dist.get_rank()}.pt")
            optim_path = os.path.join(save_dir, f"optim_world_size_{dist.get_world_size()}_rank_{dist.get_rank()}.pt")
            extra_path = os.path.join(save_dir, f"extra_state_world_size_{dist.get_world_size()}_rank_{dist.get_rank()}.pt")
            torch.save(model_state_dict, model_path)
            torch.save(optimizer_state_dict, optim_path)
            torch.save(extra_state_dict, extra_path)
        torch.distributed.barrier()

        # save for hf format
        if self.module_args.get("save_hf", True):
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            # state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            # with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, state_dict_cfg, None):
            with nullcontext():
                state_dict_config = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False)
                model_state_dict = get_model_state_dict(self.model, options=state_dict_config)
                # model_state_dict = self.model.state_dict()
                if dist.get_rank() == 0:
                    hf_path = os.path.join(save_dir, "huggingface")
                    os.makedirs(hf_path, exist_ok=True)
                    # model_config = self.model._fsdp_wrapped_module.config
                    model_config = self.model.config
                    # model_config.save_pretrained(hf_path)
                    self.model.config.save_pretrained(hf_path)
                    self.tokenizer.save_pretrained(hf_path)

                    with torch.device("meta"):
                        save_model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.bfloat16)
                    save_model.to_empty(device="cpu")
                    save_model.save_pretrained(hf_path, state_dict=model_state_dict)
            self._logger.info(f"save checkpoint to {save_dir}")

    def load_checkpoint(self, iteration):
        load_dir = f"{self.runtime_args.output_dir}/save_model/{self.name}/{iteration}"
        if not os.path.exists(load_dir):
            self._logger.info(f"{load_dir} not exists, will skip load")
            return
        model_path = os.path.join(load_dir, f"model_world_size_{dist.get_world_size()}_rank_{dist.get_rank()}.pt")
        optim_path = os.path.join(load_dir, f"optim_world_size_{dist.get_world_size()}_rank_{dist.get_rank()}.pt")
        extra_state_path = os.path.join(load_dir, f"extra_state_world_size_{dist.get_world_size()}_rank_{dist.get_rank()}.pt")

        model_state_dict = torch.load(model_path, weights_only=False)
        optimizer_state_dict = torch.load(optim_path, weights_only=False)
        extra_state_dict = torch.load(extra_state_path, weights_only=False)

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        # with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
        with nullcontext():
            self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        if "rng" in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict["rng"])
        torch.distributed.barrier()


    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        torch.cuda.set_rng_state(rng_state["cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])
    