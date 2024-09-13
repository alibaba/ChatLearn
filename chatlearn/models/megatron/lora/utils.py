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
"""Utils"""

def recursive_getattr(model, module_name):
    """
    Recursively get the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to get the attribute from.
        module_name (`str`)
            The name of the module to get the attribute from.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


def recursive_setattr(model, module_name, module):
    """
    Recursively set the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to set the attribute in.
        module_name (`str`)
            The name of the module to set the attribute in.
        module (`torch.nn.Module`)
            The module to set the attribute to.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    setattr(output, split_list[-1], module)
