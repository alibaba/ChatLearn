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
"""synchronizer"""

from chatlearn.models.megatron_module import MegatronModule
from chatlearn.models.vllm_module import VLLMModule
from chatlearn.runtime.dist_actor import DistModel
from .megatron_megatron import MegatronMegatronSync
from .megatron_vllm import MegatronVllmSync

def get_synchronizer(src_model, dst_model):
    assert isinstance(src_model, DistModel)
    assert isinstance(dst_model, DistModel)
    src_model = src_model.replicas[0].model
    dst_model = dst_model.replicas[0].model
    if isinstance(src_model, MegatronModule) and isinstance(dst_model, MegatronModule):
        return MegatronMegatronSync(src_model, dst_model)
    elif isinstance(src_model, MegatronModule) and isinstance(dst_model, VLLMModule):
        return MegatronVllmSync(src_model, dst_model)
    else:
        raise RuntimeError(f"None supported backend mapping {src_model} {dst_model}")