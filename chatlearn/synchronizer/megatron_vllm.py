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

import operator
from functools import reduce
from .base import BaseSync
from chatlearn.utils import future
from transformers import AutoConfig
from chatlearn.utils.utils import get_use_legacy_models

from chatlearn.utils.vllm_utils import Megatron2LlamaSyncMap, Megatron2QWenSyncMap, MCore2LlamaSyncMap

class MegatronVllmSync(BaseSync):

    def __init__(self, src_model, dst_model):
        super().__init__(src_model, dst_model)
        config_dir = dst_model.module_args.args_dict["tokenizer"]
        config =  AutoConfig.from_pretrained(config_dir)
        self.model_class_name = config.architectures[0]
        self.src_module_args = src_model.module_args
        self.is_parameter_changed = True

    def map_src_to_dst(self, src_names, src_pipe_layer_offset):
        """
        :meta private:
        """
        layer_offset = src_pipe_layer_offset #self.get_pipeline_layer_offset(num_src_pipeline_stage, src_pipe_stage)
        if self.model_class_name == "QWenLMHeadModel":
            sync_map_cls = Megatron2QWenSyncMap
            from chatlearn.utils.vllm_utils import fix_qwen_query_key_value_ordering # pylint: disable=import-outside-toplevel
            self._to_fix_qkv_ordering_func = fix_qwen_query_key_value_ordering
            sync_map = sync_map_cls(src_names, layer_offset, QwenVersion.v_1.value)
        elif self.model_class_name == "Qwen2ForCausalLM":
            sync_map_cls = Megatron2QWenSyncMap
            from chatlearn.utils.vllm_utils import split_attn_state
            self._to_fix_qkv_ordering_func = split_attn_state
            sync_map = sync_map_cls(src_names, layer_offset, QwenVersion.v_2.value)
        elif self.model_class_name == "LlamaForCausalLM":
            use_legacy_models = get_use_legacy_models(self.src_model.module_args.args_dict)
            sync_map_cls = Megatron2LlamaSyncMap if use_legacy_models else MCore2LlamaSyncMap
            from chatlearn.utils.vllm_utils import fix_qwen_query_key_value_ordering # pylint: disable=import-outside-toplevel
            self._to_fix_qkv_ordering_func = fix_qwen_query_key_value_ordering
            sync_map = sync_map_cls(src_names, layer_offset)
        else:
            raise RuntimeError(f"Unsupported model {type(self.model.model)}, Expect QWenLMHeadModel, Qwen2ForCausalLM or LlamaForCausalLM.")
        self.sync_map = sync_map
        self._validate(sync_map)
        # self._concat_params_dict = sync_map.concat_params_dict
        # self._to_fix_act_ordering_dict = sync_map.to_fix_act_ordering_dict
        # self._to_fix_qkv_ordering_dict = sync_map.to_fix_qkv_ordering_dict
        return sync_map.src_names, sync_map.dst_names

    def _validate(self, sync_map):
        if self.sync_map.concat_params_dict is not None:
            if isinstance(self.sync_map.concat_params_dict, dict):
                assert "modules" in self.sync_map.concat_params_dict
                assert "dim" in self.sync_map.concat_params_dict
                assert isinstance(self.sync_map.concat_params_dict["modules"], list)
            else:
                raise RuntimeError(f"Expect concat_params_dict in {self} to be a dict or None, while {self.sync_map.concat_params_dict}.")

        if self.sync_map.to_fix_act_ordering_dict is not None:
            if isinstance(self.sync_map.to_fix_act_ordering_dict, dict):
                assert "modules" in self.sync_map.to_fix_act_ordering_dict
                assert "dim" in self.sync_map.to_fix_act_ordering_dict
                assert isinstance(self.sync_map.to_fix_act_ordering_dict["modules"], list)
                fix_dim = self.sync_map.to_fix_act_ordering_dict["dim"]
            else:
                raise RuntimeError(f"Expect to_fix_act_ordering_dict in {self} to be a dict or None, while {self.sync_map.to_fix_act_ordering_dict}.")

        if self.sync_map.to_fix_qkv_ordering_dict is not None:
            if isinstance(self.sync_map.to_fix_qkv_ordering_dict, dict):
                assert "modules" in self.sync_map.to_fix_qkv_ordering_dict
                assert "layer_re" in self.sync_map.to_fix_qkv_ordering_dict
                assert isinstance(self.sync_map.to_fix_qkv_ordering_dict["modules"], list)
                # to_fix_modules_list = self.sync_map.to_fix_qkv_ordering_dict["modules"]
                #layer_re = self.sync_map.to_fix_qkv_ordering_dict["layer_re"]
            else:
                raise RuntimeError(f"Expect to_fix_qkv_ordering_dict in {self} to be a dict or None, while {self.sync_map.to_fix_qkv_ordering_dict}.")

    def map_name_from_src_to_dst(self, send_actor, recv_actor, src_names, dst_names):
        src_pipe_layer_offset = self.get_or_cache(send_actor, "get_pipeline_stage_layer_offset")
        src_names, dst_names = self.map_src_to_dst(src_names, src_pipe_layer_offset)
        # src_names, dst_names = future.get(recv_actor.map_src_to_dst.remote(src_names, src_pipe_layer_offset))
        # concat_params_dict = future.get(recv_actor.get_concat_params_dict.remote())
        # future.get(send_actor.set_concat_params_dict.remote(self.sync_map.concat_params_dict))
        # to_fix_act_ordering_dict = future.get(recv_actor.get_to_fix_act_ordering_dict.remote())
        # future.get(send_actor.set_to_fix_act_ordering_dict.remote(self.sync_map.to_fix_act_ordering_dict))
        # to_fix_qkv_ordering_dict = future.get(recv_actor.get_to_fix_qkv_ordering_dict.remote())
        # future.get(send_actor.set_to_fix_qkv_ordering_dict.remote(self.sync_map.to_fix_qkv_ordering_dict))
        # to_fix_qkv_ordering_func = future.get(recv_actor.get_to_fix_qkv_ordering_func.remote())
        # future.get(send_actor.set_to_fix_qkv_ordering_func.remote(self._to_fix_qkv_ordering_func))
        return src_names, dst_names

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
        layer_re = self.sync_map.to_fix_qkv_ordering_dict["layer_re"]
        to_fix_qkv_ordering_dict = self.sync_map.to_fix_qkv_ordering_dict
        if to_fix_qkv_ordering_dict is None:
            return params_to_sync_list
        to_fix_modules_list = to_fix_qkv_ordering_dict["modules"]
        for i, (name, params_to_sync) in enumerate(params_to_sync_list):
            from chatlearn.utils.vllm_utils import split_attn_state # pylint: disable=import-outside-toplevel
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
        for i, (name, params_to_sync) in enumerate(params_to_sync_list):
            to_fix_act_ordering_list = self.sync_map.to_fix_act_ordering_dict["modules"]
            if any([ele in name for ele in to_fix_act_ordering_list]): # pylint: disable=use-a-generator
                val = params_to_sync
                offset = val.shape[0] // 2
                w1 = val[:offset,:]
                w2 = val[offset:,:]
                params_to_sync = torch.cat([w2, w1], dim=fix_dim)
                params_to_sync_list[i] = (name, params_to_sync)
        return params_to_sync_list

    def transform_parameters(self, params_to_sync_list):
        """
        transform parameters, e.g. concat, fix ordering
        """
        params_to_sync_list = self.concat_params(params_to_sync_list)
        params_to_sync_list = self.fix_act_ordering(params_to_sync_list)
        params_to_sync_list = self.fix_qkv_ordering(params_to_sync_list)
        return params_to_sync_list

            
