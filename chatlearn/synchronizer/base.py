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


from chatlearn.utils import future
from chatlearn.utils import utils

class BaseSync:
    def __init__(self, src_model, dst_model):
        self.src_model = src_model
        self.dst_model = dst_model
        self.is_parameter_changed = False

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

    def map_name_from_src_to_dst(self, send_actor, recv_actor, src_names, dst_names):
        """
        map layer name from src model to dst model
        """

    def transform_parameters(self, params_to_sync_list):
        """
        transform parameters, e.g. concat, fix ordering
        """
        return params_to_sync_list