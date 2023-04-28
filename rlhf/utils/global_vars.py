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
"""global vars."""

_GLOBAL_ARGS = None
_EXIT_ACTOR = None


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def is_initialized():
    return _GLOBAL_ARGS.initialized


def set_initialized():
    global _GLOBAL_ARGS
    _GLOBAL_ARGS.initialized = True


def set_global_variables(args):
    """Set global vars"""
    assert args is not None
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def set_exit_actor(actor):
    global _EXIT_ACTOR
    _EXIT_ACTOR = actor