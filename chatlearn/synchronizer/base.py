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
"""base"""

import torch

from chatlearn.utils import future
from chatlearn.utils import utils

class BaseSync:
    """Base synchronizer"""

    def __init__(self, src_model, dst_model):
        self.src_model = src_model
        self.dst_model = dst_model
        self.is_parameter_changed = False
        self.concat_params_dict = None

    def get_or_cache(self, actor, func_name, *args, **kwargs):
        def inner_func(*args, **kwargs):
            return future.get(getattr(getattr(actor, func_name), 'remote')(*args, **kwargs))
        cached_name = str(actor) + "_" + func_name
        if hasattr(self, cached_name):
            cached = getattr(self, cached_name)
        else:
            cached = {}
            setattr(self, cached_name, cached)
        return utils.get_or_cache(cached, actor, inner_func, *args, **kwargs)

    def map_name_from_src_to_dst(self, send_actor, recv_actor, src_names, dst_names): # pylint: disable=unused-argument
        """
        map layer name from src model to dst model
        """
        return src_names, dst_names

    def allgather_routed_experts(self, name, params_to_sync, tp_rank, ep_rank, group_name): # pylint: disable=unused-argument
        """
        allgather routed expert params 
        """
        return params_to_sync

    def transform_parameters(self, params_to_sync_list):
        """
        transform parameters, e.g. concat, fix ordering
        """
        return params_to_sync_list

    def regroup_params_to_sync(self, name, param_data, tp_division):
        """
        :meta private:
        """
        param_data_shape = param_data.shape
        # Regroup these tensors into different tp slices.
        # Output: [tp_slice_0, tp_slice_1, ...]
        # Comment:
        #   src -> dst: [w, h * tp_size] -> tp_size * [w, h]
        #       'self_attention.dense' in QWen and LLama2 legacy
        #       'mlp.dense_4h_to_h' in QWen and LLama2 legacy model
        #       'mlp.linear_fc2' in LLama2 mcore model
        #       'mlp.shared_experts.dense_4h_to_h in QWen-MoE model
        #   src -> dst: [w * tp_size, h] -> tp_size * [w, h]
        #       'mlp.dense_h_to_4h' in QWen and LLama2 legacy
        #       'mlp.linear_fc1' in LLama2 mcore model
        #       'mlp.w1' in QWen model only for vLLM backend
        #       'mlp.shared_experts.dense_h_to_4h in QWen-MoE model
        if (
            "self_attention.dense" in name
            or "mlp.dense_4h_to_h" in name
            or "mlp.linear_fc2" in name
            or "mlp.shared_experts.dense_4h_to_h" in name
        ):
            param_data_list = []
            col_offset = param_data_shape[1] // tp_division
            for idx in range(tp_division):
                start = idx * col_offset
                end =  start + col_offset
                param_data_list.append(param_data[:,start:end])
            param_data = torch.concat(param_data_list, dim=0).view(param_data_shape)
            del param_data_list
        if (
            "mlp.dense_h_to_4h" in name
            or "mlp.linear_fc1" in name
            or ("mlp.w1" in name and self.concat_params_dict is not None)
            or "mlp.shared_experts.dense_h_to_4h" in name
        ):
            param_data_list = []
            row_offset = param_data_shape[0] // tp_division // 2
            for idx in range(tp_division):
                w1_start = idx * row_offset
                w1_end = w1_start + row_offset
                w2_start = (idx + tp_division) * row_offset
                w2_end = w2_start + row_offset
                param_data_list.append(
                    torch.concat([param_data[w1_start:w1_end,:], param_data[w2_start:w2_end,:]], dim=0))
            param_data = torch.concat(param_data_list, dim=0).view(param_data_shape)
            del param_data_list

        return param_data
