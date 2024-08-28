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
"""Megatron Hooks."""

import inspect
import importlib

megatron_exist = importlib.util.find_spec("megatron")

if megatron_exist:
    from chatlearn.utils.megatron_import_helper import initialize_megatron
    if "args_dict" not in inspect.getfullargspec(initialize_megatron).args:
        from chatlearn.models.megatron.hooks import transformer
        from chatlearn.models.megatron.hooks import generation
