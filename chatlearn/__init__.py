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
"""init"""

import importlib

from chatlearn import hooks
from chatlearn.launcher.initialize import init
from chatlearn.models.base_module import BaseModule
from chatlearn.models.megatron_module import MegatronModule
from chatlearn.models.torch_module import TorchModule
from chatlearn.models.fsdp_module import FSDPModule
from chatlearn.runtime.engine import Engine, RLHFEngine
from chatlearn.runtime.engine import Environment
from chatlearn.runtime.engine import Trainer
from chatlearn.runtime.evaluator import Evaluator
from chatlearn.runtime.model_flow import ControlDependencies
from chatlearn.utils.future import get
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger

vllm_exist = importlib.util.find_spec("vllm")
if vllm_exist:
    import vllm
    from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion  # pylint: disable=ungrouped-imports
    if CURRENT_VLLM_VERSION in [version.value for version in VLLMVersion]:
        from chatlearn.models.vllm_module import VLLMModule
    else:
        raise Exception("only vllm0.8.5 support, if you want to use previous vllm version, \
                         please git checkout 4ad5912306df5d4a814dc2dd5567fcb26f5d473b")
