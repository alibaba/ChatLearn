# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
from chatlearn.models.megatron_module import RLHFMegatronModule
from chatlearn.models.rlhf_module import RLHFModule
from chatlearn.models.torch_module import RLHFTorchModule
from chatlearn.runtime.engine import Engine
from chatlearn.runtime.engine import Environment
from chatlearn.runtime.engine import EvalEngine
from chatlearn.runtime.engine import RLHFEngine
from chatlearn.runtime.engine import Trainer
from chatlearn.runtime.evaluator import Evaluator
from chatlearn.utils.future import get
from chatlearn.utils.global_vars import get_args

vllm_exist = importlib.util.find_spec("vllm")
if vllm_exist:
    from chatlearn.models.vllm_module import RLHFVLLMModule
