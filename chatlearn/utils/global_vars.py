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
"""global vars."""

_GLOBAL_ARGS = None
_EXIT_ACTOR = None
_EXIT_ACTOR_NAME = "ChatLearnExitActor"
_DECORATED_MODELS = None
_DECORATED_OUTER_TO_INNER = {}
_DEPENDENCIES = None
_VLLM_ACTORS = None


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)

def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS

def set_global_variables(args):
    """Set global vars"""
    assert args is not None
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    global _DECORATED_MODELS
    _DECORATED_MODELS = set()

def set_decorated(model_name):
    _DECORATED_MODELS.add(model_name)

def is_decorated(model_name):
    _ensure_var_is_initialized(_DECORATED_MODELS, 'decorated_models')
    return bool(model_name in _DECORATED_MODELS)

def unwrap_func(func, level=None):
    """
    func: func to unwrap
    level: unwrap level, if level is None, unwrap to the original func
    """
    if func not in _DECORATED_OUTER_TO_INNER:
        return func
    if level is not None:
        if level > 0:
            level -= 1
        else:
            return func
    return unwrap_func(_DECORATED_OUTER_TO_INNER[func], level)

def set_wrap_func(func, new_func):
    assert new_func not in _DECORATED_OUTER_TO_INNER
    _DECORATED_OUTER_TO_INNER[new_func] = func

def set_dependencies(dependencies):
    global _DEPENDENCIES
    assert _DEPENDENCIES is None
    _DEPENDENCIES = dependencies

def reset_dependencies():
    global _DEPENDENCIES
    _DEPENDENCIES = None

def get_dependencies():
    return _DEPENDENCIES

def set_vllm_actors(actors):
    global _VLLM_ACTORS
    _VLLM_ACTORS = actors

def get_vllm_actors():
    return _VLLM_ACTORS
