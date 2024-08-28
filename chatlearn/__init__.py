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
from chatlearn.models.deepspeed_module import DeepSpeedModule
from chatlearn.models.megatron_module import MegatronModule
from chatlearn.models.torch_module import TorchModule
from chatlearn.runtime.engine import DPOEngine
from chatlearn.runtime.engine import Engine
from chatlearn.runtime.engine import Environment
from chatlearn.runtime.engine import EvalEngine
from chatlearn.runtime.engine import OnlineDPOEngine
from chatlearn.runtime.engine import GRPOEngine
from chatlearn.runtime.engine import GRPOMathEngine
from chatlearn.runtime.engine import RLHFEngine
from chatlearn.runtime.engine import Trainer
from chatlearn.runtime.evaluator import Evaluator
from chatlearn.utils.future import get
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger

vllm_exist = importlib.util.find_spec("vllm")
if vllm_exist:
    import vllm
    from chatlearn.utils.constant import CURRENT_VLLM_VERSION, VLLMVersion  # pylint: disable=ungrouped-imports
    if CURRENT_VLLM_VERSION in [version.value for version in VLLMVersion]:
        from chatlearn.models.vllm_module import VLLMModule

        # for compatibility, remove later
        class RLHFVLLMModule(VLLMModule):
            """RLHFVLLMModule is deprecated, please use VLLMModule"""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                logger.warning("RLHFVLLMModule is deprecated, please use VLLMModule")


# for compatibility, remove later
class RLHFModule(BaseModule):
    """RLHFModule is deprecated, please use BaseModule"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.warning("RLHFModule is deprecated, please use BaseModule")


# for compatibility, remove later
class RLHFTorchModule(TorchModule):
    """RLHFTorchModule is deprecated, please use TorchModule"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.warning("RLHFTorchModule is deprecated, please use TorchModule")


# for compatibility, remove later
class RLHFMegatronModule(MegatronModule):
    """RLHFMegatronModule is deprecated, please use MegatronModule"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.warning("RLHFMegatronModule is deprecated, please use MegatronModule")
