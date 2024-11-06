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
"""megatron to megatron synchronizer"""

from chatlearn.utils import future
from .base import BaseSync

class MegatronMegatronSync(BaseSync):
    """megatron to megatron synchronizer"""

    def __init__(self, src_model, dst_model):
        super().__init__(src_model, dst_model)

    def _get_dst_name(self, src_name, src_prefix, dst_prefix):
        if src_prefix:
            dst_name = src_name[len(src_prefix):]
        else:
            dst_name = dst_prefix + src_name
        return dst_name

    def set_model_prefix(self, src_names, dst_names):
        dst_prefix = None
        src_prefix = None
        for sname in src_names:
            for dname in dst_names:
                if sname in dname:
                    prefix = dname[:dname.index(sname)]
                    dst_prefix = prefix
                    return src_prefix, dst_prefix
                elif dname in sname:
                    prefix = sname[:sname.index(dname)]
                    src_prefix = prefix
                    return src_prefix, dst_prefix
        if dst_prefix is None and src_prefix is None:
            raise RuntimeError("Cannot find prefix")
        return src_prefix, dst_prefix

    def map_name_from_src_to_dst(self, send_actor, recv_actor, src_names, dst_names):
        dst_names_ref = future.get(recv_actor.get_parameter_names.remote(requires_grad=False))
        src_prefix, dst_prefix = self.set_model_prefix(src_names, dst_names_ref)
        dst_names = [self._get_dst_name(name, src_prefix, dst_prefix) for name in dst_names]
        return src_names, dst_names
