# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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
"""The mappers between architectures"""
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chatlearn.runtime.dist_actor import DistModel

def get_mapper_name(src_model: 'DistModel', dst_model: 'DistModel'):
    src_type = src_model.runtime_args.train_backend
    dst_type = dst_model.runtime_args.rollout_backend
    if src_type == 'megatron' and dst_type == 'vllm':
        return "MegatronVLLMMapper"
    elif src_type == 'megatron' and dst_type == 'sglang':
        return "MegatronSGLangMapper"
    else:
        raise NotImplementedError(f"Unsupported src/dst model combination: {src_type}-{dst_type}")


def name_to_mapper_cls(mapper_name: str):
    # pylint: disable=import-outside-toplevel
    from .mapping_helpers import VLLM_HELPERS, HF_HELPERS
    if mapper_name in ["MegatronVLLMMapper", "MegatronSGLangMapper"]:
        from .mapper import MegatronMapper
        helper_mappings = {"MegatronVLLMMapper": VLLM_HELPERS, "MegatronSGLangMapper": HF_HELPERS}
        return partial(MegatronMapper, mapper_config=helper_mappings[mapper_name])
    else:
        raise ValueError(f"Unrecognized Mapper {mapper_name}")
