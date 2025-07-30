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
"""helper function to parse datatype from Config class"""
from functools import partial
from dataclasses import fields, is_dataclass
from typing import get_origin, get_args, Dict, List, Union, Optional, Callable, Any


def parse_optional(dtype, data):
    if data is None:
        return None
    if isinstance(data, str) and data.lower() in ["none", "null"]:
        return None
    return dtype(data)

def parse_boolean(data: Union[str, bool]):
    if isinstance(data, bool):
        return data
    return data.lower() == "true"

def _resolve_type_from_dataclass(dt) -> Dict:
    """resolve parser from a dataclass object (not dataclass type!!!)"""
    if not is_dataclass(dt):
        raise ValueError("Should be a dataclass object")
    dtypes = {}
    default_parser = {bool: parse_boolean, Any: None}
    for f in fields(dt):
        dtypes[f.name] = None
        dtype = f.type if f.type is not Any else type(getattr(dt, f.name))
        # NOTE: Optional -> Union, List -> list, Union -> Union, Tuple -> tuple
        # NOTE: iteratively resolve nested dataclass
        if is_dataclass(dtype):
            dtypes[f.name] = _resolve_type_from_dataclass(getattr(dt, f.name))
        elif get_origin(dtype) is Union:
            # we only process optional here
            args = get_args(dtype)
            if len(args) == 2 and args[1] is type(None):
                parser = default_parser[args[0]] if args[0] in default_parser else args[0]
                dtypes[f.name] = partial(parse_optional, parser)
        elif dtype in default_parser:
            dtypes[f.name] = default_parser[dtype]
        elif callable(dtype):
            dtypes[f.name] = dtype
    return dtypes

def _find_type(dtypes, key: str):
    def _inner_find(dtypes, level, key_list):
        dtype = dtypes[key_list[level]]
        if isinstance(dtype, dict):
            return _inner_find(dtype, level+1, key_list)
        return dtype
    return _inner_find(dtypes, 0, key.split("."))

def find_parser_from_keyname(
    dt: Any,
    keynames: List[str]
) -> Dict[str, Optional[Callable]]:
    """Find datatype parser according to keyname by searching
    in dataclass object.

    Args:
        dt (Any): The dataclass object containing keyname.
        keynames (List[str]): The list of keynames to be parsed.

    Returns:
        Dict[str, Optional[Callable]]: mapping keyname to its parser.
        Maybe None if no parser found.
    """
    dtypes = _resolve_type_from_dataclass(dt)
    return {
        k: _find_type(dtypes, k.lstrip('+')) for k in keynames
    }
