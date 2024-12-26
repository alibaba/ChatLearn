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
"""models."""

import chatlearn # for hook purpose
from .old_policy_inference import PolicyInference
from .old_value_inference import ValueInference
from .policy_trainer import PolicyTrainer
from .value_trainer import ValueTrainer
from .reward_inference import RewardInference
from .reference import PolicyReference
try:
    from chatlearn.models.vllm import is_vllm_v2
    if is_vllm_v2():
        from .vllm_policy_inference import VLLMPolicyInferenceV2 as VLLMPolicyInference
    else:
        from .vllm_policy_inference import VLLMPolicyInference
except ImportError:
    print("Cannot import VLLMPolicyInference")
    VLLMPolicyInference = None
