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
import re
from collections import defaultdict
from abc import abstractmethod
import operator
from functools import reduce
import ray.util.collective as col
import torch
import torch.distributed as dist

from chatlearn.utils.constant import QwenVersion
from chatlearn.utils.utils import get_use_legacy_models
from chatlearn.utils.vllm_utils import fix_qwen_query_key_value_ordering
from chatlearn.utils.vllm_utils import split_attn_state
from chatlearn.utils.vllm_utils import (
    Megatron2LlamaSyncMap,
    Megatron2QWenSyncMap,
    MCore2LlamaSyncMap,
    MCore2Qwen2SyncMap,
    MCore2MoonlightSyncMap
)
from chatlearn.utils.megatron_import_memory_helper import MegatronVersion, get_megatron_version
from .base import BaseSync

class MegatronVllmSync(BaseSync):
    """Megatron to vllm sync"""

    def __init__(self, src_model, dst_model):
        super().__init__(src_model, dst_model)
        self.src_module_args = src_model.module_args
        self.dst_module_args = dst_model.module_args
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

    def allgather_routed_experts_from_hep(self, name, params_to_sync, group_name, tp_rank):
        """
        This function is applicable for synchronizing parameters from QWen with HEP enabled
        to vLLM. In HEP, routed experts are split into a total number of EP size * TP size.
        Thus, the function will all-gather across EP size * TP size routed experts and slice
        them to TP size partitions.
        """
        if self.sync_map._to_allgather_routed_experts_dict is None:
            return params_to_sync, False

        to_allgather_routed_experts_dict = self.sync_map._to_allgather_routed_experts_dict
        layer_re = to_allgather_routed_experts_dict["layer_re"]
        to_regroup_modules_list = to_allgather_routed_experts_dict["modules"]

        m = layer_re.match(name)
        if m is None:
            return params_to_sync, False

        op_name = m.group(2)
        if op_name in to_regroup_modules_list:
            tp_size = self.src_module_args.args_dict["tensor_model_parallel_size"]
            ep_size = self.src_module_args.args_dict["moe_expert_model_parallel_size"]
            hep_size = tp_size * ep_size
            moe_num_experts = self.src_module_args.args_dict["moe_num_experts"]
            local_num_experts = moe_num_experts // hep_size
            hidden_size = self.src_module_args.args_dict["hidden_size"]
            output_tensor_list = [
                torch.empty(size=params_to_sync.shape, dtype=params_to_sync.dtype, device=params_to_sync.device)
                for _ in range(hep_size)
            ]
            col.allgather(output_tensor_list, params_to_sync, group_name)
            del params_to_sync
            val_list = []
            if "dense_h_to_4h" in op_name:
                # w13_weight
                while output_tensor_list:
                    params = output_tensor_list.pop(0)
                    # regroup among difference tp slices
                    params = params.view((moe_num_experts, -1, hidden_size)).contiguous()
                    params = params.reshape((local_num_experts * 2, -1, hidden_size))
                    params = params.chunk(tp_size, dim=1)[tp_rank]
                    # reorder w1 and w3
                    params = params.reshape(params.shape[0] // 2, -1, hidden_size)
                    params_right, params_left = params.chunk(2, dim=1)
                    del params
                    params = torch.cat([params_left, params_right], dim=1)
                    del params_left
                    del params_right
                    val_list.append(params)
                params_to_sync = torch.cat(val_list, dim=0).contiguous()
            else:
                # w2_weight
                while output_tensor_list:
                    params = output_tensor_list.pop(0)
                    params = params.reshape((local_num_experts, -1, hidden_size))
                    chunked_params = params.chunk(tp_size, dim=1)[tp_rank].contiguous()
                    del params
                    val_list.append(chunked_params)
                params_to_sync = torch.cat(val_list, dim=0).transpose(1, 2).contiguous()
            del val_list
            return params_to_sync, True
        else:
            return params_to_sync, False

    def allgather_routed_experts(self, name, params_to_sync, group_name, tp_rank): # pylint: disable=unused-argument
        megatron_version = get_megatron_version()
        if megatron_version == MegatronVersion.V4:
            return self.allgather_routed_experts_from_hep(name, params_to_sync, group_name, tp_rank)
        else:
            raise NotImplementedError(
                "ChatLearn does not support all-gathering routed experts for Megatron-LM, but supports QWen with HEP enabled. "
                "Please export `QWEN_VERSION` as `qwen_moe_v1`."
            )

    def alltoall_routed_experts_from_hep(self, name, params_to_sync, comm_group):
        """
        This function is applicable for synchronizing parameters from QWen with HEP enabled
        to vLLM. In HEP, routed experts are split into a total number of EP size * TP size.
        Thus, the function will all-to-all across EP size * TP size routed experts.
        """
        if self.sync_map._to_alltoall_routed_experts_dict is None:
            return params_to_sync, False

        to_alltoall_routed_experts_dict = self.sync_map._to_alltoall_routed_experts_dict
        layer_re = to_alltoall_routed_experts_dict["layer_re"]
        to_regroup_modules_list = to_alltoall_routed_experts_dict["modules"]

        m = layer_re.match(name)
        if m is None:
            return params_to_sync, False

        op_name = m.group(2)
        if op_name in to_regroup_modules_list:
            tp_size = self.src_module_args.args_dict["tensor_model_parallel_size"]
            ep_size = self.src_module_args.args_dict["moe_expert_model_parallel_size"]
            hep_size = tp_size * ep_size
            moe_num_experts = self.src_module_args.args_dict["moe_num_experts"]

            local_num_experts = moe_num_experts // hep_size
            hidden_size = self.src_module_args.args_dict["hidden_size"]
            if "dense_h_to_4h" in op_name:
                # w13_weight
                # regroup among difference tp slices
                param = params_to_sync.view((moe_num_experts, -1, hidden_size))
                param = param.reshape((local_num_experts * 2, -1, hidden_size))
                params = list(param.chunk(hep_size, dim=1))
                # reorder w1 and w3
                params_list = []
                while params:
                    param = params.pop(0)
                    param = param.reshape(param.shape[0] // 2, -1, hidden_size)
                    param_right, param_left = param.chunk(2, dim=1)
                    del param
                    param = torch.cat([param_left, param_right], dim=1)
                    del param_left
                    del param_right
                    params_list.append(param)
                del params_to_sync
                output = [
                    torch.empty(size=params_list[i].shape, dtype=params_list[i].dtype, device=params_list[i].device)
                    for i in range(hep_size)
                ]
                torch.distributed.all_to_all(output, params_list, group=comm_group)
                del params_list
                params_to_sync = torch.cat(output, dim=0).contiguous()
                del output
            else:
                # w2_weight
                param = params_to_sync.view((local_num_experts, -1, hidden_size))
                params = list(param.chunk(hep_size, dim=1))
                params_list = [ele.contiguous() for ele in params]
                del param
                del params
                del params_to_sync
                output = [
                    torch.empty(size=params_list[i].shape, dtype=params_list[i].dtype, device=params_list[i].device)
                    for i in range(hep_size)
                ]
                torch.distributed.all_to_all(output, params_list, group=comm_group)
                del params_list
                params_to_sync = torch.cat(output, dim=0).transpose(1, 2).contiguous()
                del output
            return params_to_sync, True
        else:
            return params_to_sync, False

    def alltoall_routed_experts(self, name, params_to_sync, comm_group): # pylint: disable=unused-argument
        megatron_version = get_megatron_version()
        if megatron_version == MegatronVersion.V4:
            return self.alltoall_routed_experts_from_hep(name, params_to_sync, comm_group)
        else:
            raise NotImplementedError(
                "ChatLearn does not support all-to-all routed experts for Megatron-LM, but supports QWen with HEP enabled. "
                "Please export `QWEN_VERSION` as `qwen_moe_v1`."
            )

    def transform_parameters(self, params_to_sync_list):
        """
        transform parameters, e.g. concat, fix ordering
        """
        params_to_sync_list = self.concat_params(params_to_sync_list)
        params_to_sync_list = self.fix_act_ordering(params_to_sync_list)
        params_to_sync_list = self.fix_qkv_ordering(params_to_sync_list)
        params_to_sync_list = self.fix_shared_expert_ordering(params_to_sync_list)
        return params_to_sync_list

    def regroup_qkv_tp_slices(self, name, param_data, tp_division):
        param_data_shape = param_data.shape
        # Regroup qkv tensors into different tp slices only for inference model which enables vLLM backend.
        to_fix_qkv_ordering_dict = self.sync_map.to_fix_qkv_ordering_dict
        # pylint: disable=too-many-nested-blocks
        if ("attention.query_key_value" in name or \
                "self_attention.query_key_value" in name or \
                "self_attention.linear_qkv" in name) and 'norm' not in name:
            src_tp_size = self.src_module_args.args_dict["tensor_model_parallel_size"]
            dst_tp_size = self.dst_module_args.args_dict["tensor_model_parallel_size"]
            heads = self.src_module_args.args_dict["num_attention_heads"] // src_tp_size
            hidden_size_per_head = self.src_module_args.args_dict["hidden_size"] // self.src_module_args.args_dict["num_attention_heads"]

            param_shape = (3, heads, hidden_size_per_head) + param_data_shape[1:]
            division = reduce(operator.mul, param_shape, 1)
            num_elements = param_data.numel()
            if num_elements == division:
                if to_fix_qkv_ordering_dict is not None:
                    param_data = param_data.view(param_shape)
                    param_data_list = []
                    head_offset = heads // tp_division
                    for idx in range(tp_division):
                        start = idx * head_offset
                        end = start + head_offset
                        param_data_list.append(param_data[:,start:end])
                    param_data = torch.concat(param_data_list, dim=0).view(param_data_shape)
                    del param_data_list
            else:
                if self.src_module_args.args_dict["group_query_attention"]:
                    num_query_groups = self.src_module_args.args_dict["num_query_groups"]
                    assert num_query_groups == self.dst_module_args.args_dict["num_query_groups"], (
                        f"num_query_groups of src model ({num_query_groups}) must be equal to num_query_groups of "
                        f"dst model ({self.dst_moduel_args.args_dict['num_query_groups']}). Please double-check your config."
                    )
                    src_num_query_groups_per_replica = num_query_groups // src_tp_size
                    if dst_tp_size >= num_query_groups:
                        num_dst_kv_head_replicas = dst_tp_size // num_query_groups
                    else:
                        num_dst_kv_head_replicas = 1
                else:
                    src_num_query_groups_per_replica = heads
                    num_dst_kv_head_replicas = 1

                if to_fix_qkv_ordering_dict is not None or src_num_query_groups_per_replica == 1:
                    if len(param_data_shape) == 1:
                        param_data = param_data.view((heads + 2 * src_num_query_groups_per_replica, hidden_size_per_head))
                    else:
                        param_data = param_data.view(
                            (heads + 2 * src_num_query_groups_per_replica, hidden_size_per_head, self.src_module_args.args_dict["hidden_size"]))
                    param_data_list = []
                    head_offset = heads // tp_division
                    for idx in range(tp_division):
                        q_start = idx * head_offset
                        q_end = q_start + head_offset
                        if num_dst_kv_head_replicas == 1:
                            if src_num_query_groups_per_replica > tp_division:
                                assert src_num_query_groups_per_replica % tp_division == 0, (
                                    f"num_query_groups per replica of src model ({src_num_query_groups_per_replica}) "
                                    f"must be divisible by tp_division ({tp_division}). Please double-check your config."
                                )
                                kv_offset = src_num_query_groups_per_replica // tp_division
                            else:
                                kv_offset = 1
                            k_start = (heads + idx) if src_num_query_groups_per_replica // tp_division else heads
                            k_end = k_start + kv_offset
                            v_start = k_start + src_num_query_groups_per_replica
                            v_end = v_start + kv_offset
                        else:
                            k_start = heads + idx // num_dst_kv_head_replicas
                            k_end = k_start + 1
                            v_start = k_start + src_num_query_groups_per_replica
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

    def regroup_params_to_sync(self, name, param_data, tp_division, regroup_routed_experts=False):
        param_data = self.regroup_qkv_tp_slices(name, param_data, tp_division)
        return super().regroup_params_to_sync(name, param_data, tp_division, regroup_routed_experts)

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

class MegatronVllmQWen2MCoreSync(MegatronVllmSync):
    """qwen2-dense-mcore"""

    def map_src_to_dst(self, src_names, src_pipe_layer_offset):
        self._to_fix_qkv_ordering_func = split_attn_state
        return MCore2Qwen2SyncMap(src_names, src_pipe_layer_offset)

class MegatronVllmLlamaSync(MegatronVllmSync):
    """llama"""

    def map_src_to_dst(self, src_names, src_pipe_layer_offset):
        use_legacy_models = get_use_legacy_models(self.src_model.module_args.args_dict)
        sync_map_cls = Megatron2LlamaSyncMap if use_legacy_models else MCore2LlamaSyncMap
        self._to_fix_qkv_ordering_func = fix_qwen_query_key_value_ordering
        return sync_map_cls(src_names, src_pipe_layer_offset)


class MegatronVllmMoonlightSync(MegatronVllmSync):
    """Moonlight"""

    def stack_group_gemm(self, params_to_sync_list):
        """
            Currently, VLLM use FusedMOE in the MoE layer, whose gemm has the shape:
              w13: (num_experts, 2 * intermediate_size, hidden_size)
              w2: (num_experts, hidden_size,, intermediate_size)
            
            However, in TEGroupMLP, the gemm weights is split by expert like follows:
              linear_fc1.weight{i} for i in range(n_experts): (2 * intermediate_size, hidden_size)
              linear_fc2.weight{i} for i in range(n_experts): (hidden_size, intermediate_size)

            Futhermore, to bypass EP division, a workaround is applied to these weights.
            linear_fc1.weight{i} for i in range(n_experts // EP): (EP, 2 * intermediate_size, hidden_size)
            linear_fc2.weight{i} for i in range(n_experts // EP): (EP, hidden_size, intermediate_size)
        
            Therefore, we need to regroup and stack the gemm weights by expert when TEGroupMLP is used.
        """
        layer_re = re.compile(r"(.*weight)([0-9]+)")
        stack_dict = defaultdict(dict)
        params_to_sync_list_new = []
        for i, (name, params_to_sync) in enumerate(params_to_sync_list):
            m = layer_re.match(name)
            if m is None:
                params_to_sync_list_new.append((name, params_to_sync))
                continue
            key, expert_id = m.group(1), m.group(2)
            if key not in stack_dict:
                params_to_sync_list_new.append([key, None]) # placeholder
            stack_dict[key][int(expert_id)] = params_to_sync

        for i, (name, params_to_sync) in enumerate(params_to_sync_list_new):
            if params_to_sync is None:
                datas = stack_dict[name]
                num_experts = sum(data.shape[0] for data in datas.values())
                weights = []
                for expert_id in range(num_experts):
                    weight_id = expert_id % len(datas)
                    ep_rank = expert_id // len(datas)
                    weights.append(datas[weight_id][ep_rank])
                params_to_sync = torch.stack(weights, dim=0)
                params_to_sync_list_new[i] = (name, params_to_sync)
                stack_dict[name] = None

        return params_to_sync_list_new

    def collect_linear_kv_down_proj(self, params_to_sync_list):
        """
            Megatron-Core applies ColumnLinear in each proj of MLASelfAttention.
            However, kv_a_proj_with_mqa is a ReplicatedLinear instead of ColumnLinear in vLLM.

            This function do an all gather on an tp-split linear to collect full params across tp group.
        """

        from megatron.core.parallel_state import ( # pylint: disable=import-outside-toplevel
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_world_size,
        )

        to_be_merged = []
        for i, (name, params_to_sync) in enumerate(params_to_sync_list):
            if 'linear_kv_down_proj' in name or 'linear_q_down_proj' in name:
                to_be_merged.append([i, name, params_to_sync])
        to_be_merged = sorted(to_be_merged, key=lambda x: x[1])
        for idx, name, params_to_sync in to_be_merged:
            w, h = params_to_sync.shape
            out_tensor = torch.empty(
                [w * get_tensor_model_parallel_world_size(), h],
                dtype=params_to_sync.dtype,
                device=params_to_sync.device
            )
            dist.all_gather_into_tensor(out_tensor, params_to_sync, group=get_tensor_model_parallel_group())
            params_to_sync_list[idx] = (name, out_tensor)
        return params_to_sync_list

    def transform_parameters(self, params_to_sync_list):
        params_to_sync_list = super().transform_parameters(params_to_sync_list)
        params_to_sync_list = self.stack_group_gemm(params_to_sync_list)
        params_to_sync_list = self.collect_linear_kv_down_proj(params_to_sync_list)
        return params_to_sync_list

    def map_src_to_dst(self, src_names, src_pipe_layer_offset):
        self._to_fix_qkv_ordering_func = split_attn_state
        return MCore2MoonlightSyncMap(src_names, src_pipe_layer_offset)
