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
"""megatron to vllm synchronizer"""

from abc import abstractmethod
import operator
from functools import reduce
import ray.util.collective as col
import torch
from chatlearn.utils.constant import QwenVersion
from chatlearn.utils.utils import get_use_legacy_models
from chatlearn.utils.vllm_utils import fix_qwen_query_key_value_ordering
from chatlearn.utils.vllm_utils import split_attn_state
from chatlearn.utils.vllm_utils import Megatron2LlamaSyncMap, Megatron2QWenSyncMap, MCore2LlamaSyncMap
from .base import BaseSync

class MegatronVllmSync(BaseSync):
    """Megatron to vllm sync"""

    def __init__(self, src_model, dst_model):
        super().__init__(src_model, dst_model)
        self.src_module_args = src_model.module_args
        self.is_parameter_changed = True

    @abstractmethod
    def map_src_to_dst(self, src_names, src_pipe_layer_offset):
        """
        :meta private:
        """

    def _validate(self, sync_map):
        if sync_map.concat_params_dict is not None:
            if isinstance(sync_map.concat_params_dict, dict):
                assert "modules" in sync_map.concat_params_dict
                assert "dim" in sync_map.concat_params_dict
                assert isinstance(sync_map.concat_params_dict["modules"], list)
            else:
                raise RuntimeError(f"Expect concat_params_dict in {self} to be a dict or None, while {sync_map.concat_params_dict}.")

        if sync_map.to_fix_act_ordering_dict is not None:
            if isinstance(sync_map.to_fix_act_ordering_dict, dict):
                assert "modules" in sync_map.to_fix_act_ordering_dict
                assert "dim" in sync_map.to_fix_act_ordering_dict
                assert isinstance(sync_map.to_fix_act_ordering_dict["modules"], list)
            else:
                raise RuntimeError(f"Expect to_fix_act_ordering_dict in {self} to be a dict or None, while {sync_map.to_fix_act_ordering_dict}.")

        if sync_map.to_fix_qkv_ordering_dict is not None:
            if isinstance(sync_map.to_fix_qkv_ordering_dict, dict):
                assert "modules" in sync_map.to_fix_qkv_ordering_dict
                assert "layer_re" in sync_map.to_fix_qkv_ordering_dict
                assert isinstance(sync_map.to_fix_qkv_ordering_dict["modules"], list)
            else:
                raise RuntimeError(f"Expect to_fix_qkv_ordering_dict in {self} to be a dict or None, while {sync_map.to_fix_qkv_ordering_dict}.")

    def map_name_from_src_to_dst(self, send_actor, recv_actor, src_names, dst_names):
        src_pipe_layer_offset = self.get_or_cache(send_actor, "get_pipeline_stage_layer_offset")
        dst_pipe_layer_offset = self.get_or_cache(recv_actor, "get_pipeline_stage_layer_offset")
        self.sync_map = self.map_src_to_dst(src_names, src_pipe_layer_offset+dst_pipe_layer_offset)
        self._validate(self.sync_map)
        self.concat_params_dict = self.sync_map.concat_params_dict
        return self.sync_map.src_names, self.sync_map.dst_names

    def concat_params(self, params_to_sync_list):
        if self.sync_map.concat_params_dict is None:
            return params_to_sync_list
        concat_modules_list = self.sync_map.concat_params_dict["modules"]
        concat_dim = self.sync_map.concat_params_dict["dim"]
        params_to_sync_list_new = []
        concat = []
        for name, params in params_to_sync_list:
            if any(ele in name for ele in concat_modules_list):
                concat.append(params)
                if len(concat) == len(concat_modules_list):
                    params = torch.cat(concat, dim=concat_dim)
                    params_to_sync_list_new.append((name, params))
                    concat = []
            else:
                params_to_sync_list_new.append((name, params))
        return params_to_sync_list_new

    def fix_qkv_ordering(self, params_to_sync_list):
        to_fix_qkv_ordering_dict = self.sync_map.to_fix_qkv_ordering_dict
        if to_fix_qkv_ordering_dict is None:
            return params_to_sync_list
        layer_re = self.sync_map.to_fix_qkv_ordering_dict["layer_re"]
        to_fix_modules_list = to_fix_qkv_ordering_dict["modules"]
        for i, (name, params_to_sync) in enumerate(params_to_sync_list):
            m = layer_re.match(name)
            if m is None:
                continue
            op_name = m.group(2)
            if op_name in to_fix_modules_list:
                checkpoint_version = 3.0
                tp_size = self.src_module_args.args_dict["tensor_model_parallel_size"]
                heads = self.src_module_args.args_dict["num_attention_heads"] // tp_size
                hidden_size_per_head =  self.src_module_args.args_dict["hidden_size"] // self.src_module_args.args_dict["num_attention_heads"]
                if self._to_fix_qkv_ordering_func is split_attn_state:
                    _num_query_groups = self.src_module_args.args_dict["num_query_groups"]//tp_size  \
                        if self.src_module_args.args_dict["group_query_attention"] else heads
                    params_to_sync = self._to_fix_qkv_ordering_func(
                        params_to_sync, heads, _num_query_groups, hidden_size_per_head, self.src_module_args.args_dict["hidden_size"])
                    params_to_sync_list[i] = (name, params_to_sync)
                else:
                    input_shape = params_to_sync.size()
                    shape = (heads, hidden_size_per_head, 3) + input_shape[1:]
                    division = reduce(operator.mul, shape, 1)
                    num_elements = params_to_sync.numel()
                    if num_elements == division:
                        # model with gqa dont need to fix qkv ordering.
                        weight_or_bias = m.group(3)
                        params_to_sync = self._to_fix_qkv_ordering_func(
                            params_to_sync, checkpoint_version, 3, heads, hidden_size_per_head
                        )
                        if weight_or_bias == "weight":
                            params_to_sync = params_to_sync.contiguous()
                        params_to_sync_list[i] = (name, params_to_sync)
        return params_to_sync_list

    def fix_act_ordering(self, params_to_sync_list):
        if self.sync_map.to_fix_act_ordering_dict is None:
            return params_to_sync_list
        fix_dim = self.sync_map.to_fix_act_ordering_dict["dim"]
        to_fix_act_ordering_list = self.sync_map.to_fix_act_ordering_dict["modules"]
        for i, (name, params_to_sync) in enumerate(params_to_sync_list):
            if any([ele in name for ele in to_fix_act_ordering_list]): # pylint: disable=use-a-generator
                val = params_to_sync
                offset = val.shape[0] // 2
                w1 = val[:offset,:]
                w2 = val[offset:,:]
                params_to_sync = torch.cat([w2, w1], dim=fix_dim)
                params_to_sync_list[i] = (name, params_to_sync)
        return params_to_sync_list

    def fix_shared_expert_ordering(self, params_to_sync_list):
        if self.sync_map.to_fix_shared_expert_ordering is None:
            return params_to_sync_list
        fix_dim = self.sync_map.to_fix_shared_expert_ordering["dim"]
        to_fix_shared_expert_ordering_list = self.sync_map.to_fix_shared_expert_ordering["modules"]
        for i, (name, params_to_sync) in enumerate(params_to_sync_list):
            if any([ele in name for ele in to_fix_shared_expert_ordering_list]): # pylint: disable=use-a-generator
                w1, w2 = params_to_sync.chunk(2, dim=0)
                params_to_sync = torch.cat([w2, w1], dim=fix_dim).contiguous()
                params_to_sync_list[i] = (name, params_to_sync)
        return params_to_sync_list

    def regroup_routed_experts(self, name, params_to_sync, tp_rank, ep_rank, group_name):
        to_regroup_routed_experts_dict = self.sync_map.to_regroup_routed_experts_dict
        if to_regroup_routed_experts_dict is None:
            return params_to_sync_list
        layer_re = self.sync_map.to_regroup_routed_experts_dict["layer_re"]
        to_regroup_modules_list = to_regroup_routed_experts_dict["modules"]

        m = layer_re.match(name)
        if m is not None:
            op_name = m.group(2)
            if op_name in to_regroup_modules_list:
                if "dense_h_to_4h" in op_name:
                    # w13_weight
                    tp_size = self.src_module_args.args_dict["tensor_model_parallel_size"]
                    moe_num_experts = self.src_module_args.args_dict["moe_num_experts"]
                    hidden_size = self.src_module_args.args_dict["hidden_size"]
                    output_tensor_list = [
                        torch.empty(size=params_to_sync.shape, dtype=params_to_sync.dtype, device=params_to_sync.device) \
                        for _ in range(tp_size)
                    ]
                    col.allgather(output_tensor_list, params_to_sync, group_name)
                    val_list = []
                    for params in output_tensor_list:
                        params = params.view((moe_num_experts, -1, hidden_size)).contiguous()
                        params = params.reshape((moe_num_experts // tp_size * 2, -1, hidden_size))
                        params = params.chunk(tp_size, dim=1)[tp_rank]
                        params = params.reshape(params.shape[0] // tp_size * 2, -1, hidden_size)
                        params_right, params_left = params.chunk(2, dim=1)
                        params = torch.cat([params_left, params_right], dim=1)
                        val_list.append(params)
                    params_to_sync = torch.cat(val_list, dim=0).contiguous()
                else:
                    # w2_weight
                    tp_size = self.src_module_args.args_dict["tensor_model_parallel_size"]
                    moe_num_experts = self.src_module_args.args_dict["moe_num_experts"]
                    hidden_size = self.src_module_args.args_dict["hidden_size"]
                    output_tensor_list = [
                        torch.empty(size=params_to_sync.shape, dtype=params_to_sync.dtype, device=params_to_sync.device) \
                        for _ in range(tp_size)
                    ]
                    col.allgather(output_tensor_list, params_to_sync, group_name)
                    val_list = []
                    for params in output_tensor_list:
                        params = params.reshape((moe_num_experts // tp_size, -1, hidden_size))
                        params = params.chunk(tp_size, dim=1)[tp_rank]
                        val_list.append(params)
                    params_to_sync = torch.cat(val_list, dim=0).transpose(1, 2).contiguous()
                return params_to_sync, True
        return params_to_sync, False

    def transform_parameters(self, params_to_sync_list):
        """
        transform parameters, e.g. concat, fix ordering
        """
        params_to_sync_list = self.concat_params(params_to_sync_list)
        params_to_sync_list = self.fix_act_ordering(params_to_sync_list)
        params_to_sync_list = self.fix_qkv_ordering(params_to_sync_list)
        params_to_sync_list = self.fix_shared_expert_ordering(params_to_sync_list)
        return params_to_sync_list

    def regroup_qkv_tp_slices(self, name, param_data, tp_divition):
        param_data_shape = param_data.shape
        # Regroup qkv tensors into different tp slices only for inference model which enables vLLM backend.
        to_fix_qkv_ordering_dict = self.sync_map.to_fix_qkv_ordering_dict
        if "attention.query_key_value" in name or \
                "self_attention.query_key_value" in name or \
                "self_attention.linear_qkv" in name:
            tp_size = self.src_module_args.args_dict["tensor_model_parallel_size"]
            heads = self.src_module_args.args_dict["num_attention_heads"] // tp_size
            hidden_size_per_head = self.src_module_args.args_dict["hidden_size"] // self.src_module_args.args_dict["num_attention_heads"]

            param_shape = (3, heads, hidden_size_per_head) + param_data_shape[1:]
            division = reduce(operator.mul, param_shape, 1)
            num_elements = param_data.numel()
            if num_elements == division:
                if to_fix_qkv_ordering_dict is not None:
                    param_data = param_data.view(param_shape)
                    param_data_list = []
                    head_offset = heads // tp_divition
                    for idx in range(tp_divition):
                        start = idx * head_offset
                        end = start + head_offset
                        param_data_list.append(param_data[:,start:end])
                    param_data = torch.concat(param_data_list, dim=0).view(param_data_shape)
                    del param_data_list
            else:
                _num_query_groups = self.src_module_args.args_dict["num_query_groups"]//tp_size  \
                    if self.src_module_args.args_dict["group_query_attention"] else heads
                if to_fix_qkv_ordering_dict is not None or _num_query_groups == 1:
                    if len(param_data_shape) == 1:
                        param_data = param_data.view((heads + 2 * _num_query_groups, hidden_size_per_head))
                    else:
                        param_data = param_data.view(
                            (heads + 2 * _num_query_groups, hidden_size_per_head, self.src_module_args.args_dict["hidden_size"]))
                    param_data_list = []
                    head_offset = heads // tp_divition
                    for idx in range(tp_divition):
                        q_start = idx * head_offset
                        q_end = q_start + head_offset
                        k_start = (heads + idx) if _num_query_groups // tp_divition else heads
                        k_end = k_start + 1
                        v_start = k_start + _num_query_groups
                        v_end = v_start + 1

                        q_proj = param_data[q_start:q_end].contiguous()
                        k_proj = param_data[k_start:k_end].contiguous()
                        v_proj = param_data[v_start:v_end].contiguous()

                        qkv_proj = torch.cat([q_proj, k_proj, v_proj], dim=0)

                        if len(param_data_shape) == 1:
                            qkv_proj = qkv_proj.reshape(-1).contiguous()
                        else:
                            qkv_proj = qkv_proj.reshape(-1, self.src_module_args.args_dict["hidden_size"]).contiguous()

                        param_data_list.append(qkv_proj)
                    param_data = torch.concat(param_data_list, dim=0)
                    del param_data_list
        return param_data

    def regroup_params_to_sync(self, name, param_data, tp_division):
        param_data = self.regroup_qkv_tp_slices(name, param_data, tp_division)
        return super().regroup_params_to_sync(name, param_data, tp_division)

class MegatronVllmQWenSync(MegatronVllmSync):
    """qwen"""

    def map_src_to_dst(self, src_names, src_pipe_layer_offset):
        """
        :meta private:
        """
        self._to_fix_qkv_ordering_func = fix_qwen_query_key_value_ordering
        return Megatron2QWenSyncMap(src_names, src_pipe_layer_offset, QwenVersion.v_1.value)


class MegatronVllmQWen2Sync(MegatronVllmSync):
    """qwen2"""

    def map_src_to_dst(self, src_names, src_pipe_layer_offset):
        self._to_fix_qkv_ordering_func = split_attn_state
        return Megatron2QWenSyncMap(src_names, src_pipe_layer_offset, QwenVersion.v_2.value)


class MegatronVllmLlamaSync(MegatronVllmSync):
    """llama"""

    def map_src_to_dst(self, src_names, src_pipe_layer_offset):
        use_legacy_models = get_use_legacy_models(self.src_model.module_args.args_dict)
        sync_map_cls = Megatron2LlamaSyncMap if use_legacy_models else MCore2LlamaSyncMap
        self._to_fix_qkv_ordering_func = fix_qwen_query_key_value_ordering
        return sync_map_cls(src_names, src_pipe_layer_offset)
