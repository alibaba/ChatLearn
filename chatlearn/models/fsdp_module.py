# pylint: disable=import-outside-toplevel
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
"""FSDP module"""
import os
import random
import gc
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch import optim, nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict, get_model_state_dict
from torch.multiprocessing.reductions import reduce_tensor
from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from chatlearn.utils.logger import debug_rank_0
from chatlearn.utils.utils import dict_to_simplenamespace
from chatlearn.utils.communication_op import set_sp_parallel_group
from chatlearn.models.patches.monkey_patch import apply_sp_monkey_patch, apply_group_gemm
from chatlearn.runtime.decorator import timeit, monitor_error
from .torch_module import TorchModule


class FSDPModule(TorchModule):
    """TorchModule is the class for Alignment Torch models.

    Args
    ----
    name : str
        model name
    """

    def __init__(self, name: str, args=None, replica_id: int=0):
        """The chatlearn wrapper for a FSDP model.

        Args:
            name (str): The name of this module
            args (Any, optional): The arguments. Defaults to None.
            replica_id (int, optional): The replica id of this module. Defaults to 0.
        """
        super().__init__(name, args=args, replica_id=replica_id)

        self.fsdp_size = self.module_args.fsdp_size
        self.sp_size = self.module_args.ulysses_sequence_parallel_size
        self.device_mesh = None
        self.sp_device_mesh = None
        self.packing = self.module_args.packing
        self.max_token_in_seq = self.module_args.max_token_in_packing
        self.generate_micro_batch_size = self.module_args.generation_batch_size
        if self.module_args.trainable:
            self.train_micro_batch_size = self.module_args.train_micro_batch_size

    @staticmethod
    def init_fn(x: torch.nn.Module):
        if torch.distributed.get_rank() != 0:
            x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
            torch.cuda.empty_cache()
        return x

    def fsdp2_clip_grad_norm_(self, parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
        # TODO: support partial parameters FSDP2 warp
        assert norm_type==2.0, "only support l2 grad norm"

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        else:
            # prevent generators from being exhausted
            parameters = list(parameters)
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
        # manual reduce grad norm
        total_norm = total_norm.to_local()
        total_norm = total_norm ** norm_type
        dist.all_reduce(total_norm, group=self.device_mesh.get_group("fsdp"), op=torch.distributed.ReduceOp.SUM)
        total_norm = total_norm ** (1.0 / norm_type)
        _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)

        return total_norm

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

    def check_sp_compatibility(self, config):
        assert config.num_attention_heads % self.sp_size == 0, \
            "num_attention_heads must be divisible by sp"
        if self.sp_size > config.num_key_value_heads:
            assert self.sp_size % config.num_key_value_heads == 0, \
                "When sp_size > num_key_value_heads, sp_size must be divisible by num_key_value_heads"

    @monitor_error()
    @timeit()
    def model_setup(self):
        """
        :meta private:
        """
        if self.module_args.use_expandable_segments:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
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

        # get state_dict to init model for meta init
        full_state = None
        if self.module_args.meta_init:
            full_state = model.state_dict()

        # fsdp2 warp
        mix_precision_config = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)
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
            if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or \
                (isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings):
                modules.append(module)

        for module in modules:
            fully_shard(module, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)
        if self.module_args.meta_init:
            # save buffer data
            buffer_dict = {}
            for name, buf in model.named_buffers():
                buffer_dict[name] = buf
            model.to_empty(device="cuda")

            # load real state dict
            options = StateDictOptions(full_state_dict=True, cpu_offload=False, broadcast_from_rank0=True)

            # module-wise sync avoid OOM while run model like qwen3-moe-235B
            for name, module in model.named_modules():
                has_weights = any(k.startswith(name + ".") for k in full_state.keys()) and len(list(module.children()))==0
                if has_weights:
                    set_model_state_dict(
                        module,
                        {k.replace(name + ".", ""): v for k, v in full_state.items() if k.startswith(name + ".")},
                        options=options
                    )
            # set_model_state_dict(model, full_state, options=options)

            # load buffer data
            if dist.get_rank()==0:
                for name, buf in model.named_buffers():
                    buf.data.copy_(buffer_dict[name])
            torch.cuda.synchronize()
            for name, buf in model.named_buffers():
                dist.broadcast(buf, src=0)

        self.model = model
        self.model.to(torch.float32)

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
        del full_state
        self.offload()

    def get_fsdp_param_name(self, block_size=3_000_000_000) -> List[List]:
        name_list = []
        param_cnt = 0
        current_group = []
        for name, param in self.model.named_parameters():
            param_cnt += param.numel()
            current_group.append(name)
            if param_cnt >= block_size:
                name_list.append(current_group)
                current_group = []
                param_cnt = 0
        if len(current_group) > 0:
            name_list.append(current_group)
        return name_list

    def get_weight_ipc_handles_by_name(self, block_name: List[str]):
        """
        get fsdp warpped module weight by name get from named_parameters
        avoid get total model state_dict
        """
        rollout_engine = self._runtime_args.rollout_backend
        if rollout_engine == "sglang":
            # lazy import sglang
            from sglang.srt.utils import MultiprocessingSerializer
            from sglang.srt.patch_torch import monkey_patch_torch_reductions
            monkey_patch_torch_reductions()
        if self.module_args.use_expandable_segments:
            torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        reduce_tensor_dict = {}
        serialize_func = reduce_tensor if rollout_engine=='vllm' else MultiprocessingSerializer.serialize
        for name, param in self.model.named_parameters():
            if name in block_name:
                reduce_tensor_dict[name] = serialize_func(param.full_tensor().detach() \
                                        if isinstance(param, DTensor) else param.detach())
        if self.module_args.use_expandable_segments:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        return reduce_tensor_dict

    def update_weights_from_buckets(self, buckets):
        pass

    @torch.no_grad()
    def onload_weights(self, empty_cache=True):
        device_id = torch.cuda.current_device()
        self.model.to(torch.device(f"cuda:{device_id}"))
        if empty_cache:
            gc.collect()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def offload_weights(self, empty_cache=True):
        self.model.cpu()
        torch.cuda.ipc_collect()
        if empty_cache:
            gc.collect()
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
        torch.cuda.synchronize()
        if empty_cache:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def onload_optimizer_states(self, empty_cache=True):
        if not self.optimizer.state:
            return
        device_id = torch.cuda.current_device()
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(torch.device(f"cuda:{device_id}"), non_blocking=True)

        if empty_cache:
            torch.cuda.empty_cache()

    @timeit()
    def save_checkpoint(self, iteration):
        save_dir = f"{self.runtime_args.output_dir}/save_model/{self.name}/{iteration}"
        if dist.get_rank() == 0 and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Make sure directory exists before writing
        dist.barrier()

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

            state_dict_config = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False)
            model_state_dict = get_model_state_dict(self.model, options=state_dict_config)
            if dist.get_rank() == 0:
                hf_path = os.path.join(save_dir, "huggingface")
                os.makedirs(hf_path, exist_ok=True)
                model_config = self.model.config
                model_config.save_pretrained(hf_path)
                self.tokenizer.save_pretrained(hf_path)

                with torch.device("meta"):
                    save_model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")
                save_model.save_pretrained(hf_path, state_dict=model_state_dict)
            torch.distributed.barrier()
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
    