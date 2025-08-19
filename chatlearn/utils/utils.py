# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""utils"""

import ast
import inspect
import socket
import subprocess
import textwrap
import time
import copy
from contextlib import closing
from types import SimpleNamespace
from typing import Dict, List, Union, Any

import pynvml
import numpy as np
import torch
from chatlearn.utils.logger import logger

try:
    from packaging.version import Version as PkgVersion

    HAVE_PACKAGING = True
except ImportError:
    HAVE_PACKAGING = False
_te_version = None

def get_attributes(cls):
    """Get attributes from class."""
    return [(name, attr) for name, attr in inspect.getmembers(cls)
            if not (name.startswith('_')) and (not callable(attr))]


def parse_function_args(func):
    args = []

    def parse_func_args(node):
        for argument in node.args.args:
            args.append(argument.arg)

    node_iter = ast.NodeVisitor()
    node_iter.visit_FunctionDef = parse_func_args
    code = textwrap.dedent(inspect.getsource(func))
    node_iter.visit(ast.parse(code))
    return args


def get_return_lines(node):
    for line in node.body:
        if isinstance(line, ast.Return):
            return line
    for line in node.body:
        if isinstance(line, ast.If):
            return get_return_lines(line)


def get_return_value_num(ret):
    if isinstance(ret.value, ast.Name):
        return 1
    elif isinstance(ret.value, ast.Tuple):
        return len(ret.value.elts)
    elif isinstance(ret.value, ast.Call):
        raise RuntimeError("current do not support nested call in return")


def parse_function_return_num(func):
    results = []

    def parse_func_return(node):
        ret = get_return_lines(node)
        return_num = 0
        if ret is not None:
            return_num = get_return_value_num(ret)
        results.append(return_num)

    node_iter = ast.NodeVisitor()
    node_iter.visit_FunctionDef = parse_func_return
    code = textwrap.dedent(inspect.getsource(func))
    node_iter.visit(ast.parse(code))
    return results[0]


def get_host_addr():
    """
    get ip address in current node
    """
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    return ip_addr


def get_free_port():
    """
    find a free port
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as handle:
        handle.bind(('', 0))
        handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return handle.getsockname()[1]


def split_index(length, num_splits):
    # Calculate the size of each split
    size = length // num_splits
    remainder = length % num_splits

    # Initialize an empty list for indices
    indices = []

    # Loop over the number of splits and append indices
    start = 0
    end = 0
    for _ in range(num_splits):
        end += size
        if remainder > 0:
            end += 1
            remainder -= 1
        indices.append((start, end))
        start = end

    # Return the list of indices
    return indices


def to_device(device, args):
    """
    Convert args to device recursively

    Args:
        device: gpu/cpu
        args: args to be converted
    """
    if isinstance(args, (list, tuple)):
        args = type(args)(to_device(device, arg) for arg in args)
    elif isinstance(args, dict):
        for key, value in args.items():
            args[key] = to_device(device, value)
    elif isinstance(args, torch.Tensor):
        args = args.to(device)
    return args


def get_or_cache(cache, key, func, *args, **kwargs):
    """
    get results if cached
    otherwise call the func to get the results, and cache the results
    """
    if key in cache:
        return cache[key]
    res = func(*args, **kwargs)
    cache[key] = res
    return res


def flatten(nested_list):
    flat = []
    for elem in nested_list:
        if isinstance(elem, list):
            flat.extend(flatten(elem))
        else:
            flat.append(elem)
    return flat


def get_indent_count(string):
    count = 0
    for s in string:
        if s == ' ':
            count += 1
        else:
            return count


def detect_and_insert_code(lines, pattern, new_code, additional_indent=0, line_offset=0, replace=False):
    """
    Insert new_code above the pattern detected
    """
    detected_lines = [(line_number, line) for line_number, line in enumerate(lines) if pattern in line]
    if not detected_lines:
        return
    type_line_number, type_line = detected_lines[0]
    indent = get_indent_count(type_line) + additional_indent
    new_lines = [line for line in new_code.split('\n') if line.strip()]
    added_lines = []
    for line in new_lines:
        added_lines.append(" "*indent + line)
    lines = lines[:type_line_number+line_offset - replace] + added_lines + lines[type_line_number+line_offset:]
    return lines

def detect_and_insert_code_to_func(source_code, pattern, new_code, additional_indent=0, line_offset=0, replace=False):
    lines = source_code.split('\n')
    lines = detect_and_insert_code(lines, pattern, new_code, additional_indent, line_offset, replace)
    if lines is None:
        return
    indent = get_indent_count(lines[0])
    lines = [line[indent:] for line in lines]
    return '\n'.join(lines)

def execute(cmd, check=False, retry=1):
    """
    Execute cmd in shell
    
    Args:
        check: if returncode is non-zero, raise error
    """
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
    state = ret.returncode == 0
    msg = ret.stdout if state else ret.stderr
    if not state:
        logger.warning(f"execute {cmd} got error {msg}")
        if retry > 1:
            logger.warning(f"retry {cmd} ...")
            time.sleep(1)
            return execute(cmd, check, retry-1)
    return state, msg


def is_connection_refused(msg):
    keywords = ["StatusCode.UNAVAILABLE", "Connection refused", "failed to connect to all addresses"]
    return any(keyword in msg for keyword in keywords)


def get_ray_status():
    cluster_state, msg = execute("ray status", retry=3)
    if cluster_state:
        return True, None
    elif is_connection_refused(msg):
        return False, msg
    # unknown msg
    return True, msg


def get_full_proc_memory_info(prefix):
    torch.cuda.synchronize()
    s = prefix + ': '
    s += f'memory allocated: {torch.cuda.memory_allocated() / (1 << 30):.2f} GiB, ' \
         f'memory reserved: {torch.cuda.memory_reserved() / (1 << 30):.2f} GiB, ' \
         f'proc memory usage: {nvml_proc_memory_info()}'
    return s


def nvml_proc_memory_info():
    pynvml.nvmlInit()
    s = ''
    for dev_id in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        mem_str = ' | '.join([f'(pid {proc.pid}: {proc.usedGpuMemory / (1 << 30):.2f} GiB)' \
                  for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)])
        s += mem_str
        break
    return s


def dict_to_simplenamespace(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_simplenamespace(value)
    return SimpleNamespace(**d)

def regroup_by_concat_along_batch(data: List[Dict[str, Union[torch.Tensor, List[Any]]]]) -> Dict[str, Union[torch.Tensor, List[Any]]]:
    """
    Merge a List[Dict] in to one Dict
    """
    batched = {}
    if data[0] is None:
        return batched
    for key in data[0].keys():
        to_batch = [results[key] for results in data]
        if isinstance(to_batch[0], torch.Tensor):
            if len(to_batch[0].shape) == 2:
                max_dim_1 = max([ele.shape[1] for ele in to_batch]) # pylint: disable=consider-using-generator
                pad_value = 0.0 if to_batch[0].dtype in [torch.float32, torch.float16, torch.bfloat16] else 0
                value = [
                    torch.nn.functional.pad(
                        ele,
                        (0, max_dim_1 - ele.shape[1]),
                        value=pad_value,
                    )
                    for ele in to_batch
                ]
                batched[key] = torch.vstack(value)
            elif len(to_batch[0].shape) == 1:
                batched[key] = torch.concat(to_batch)
            else:
                raise RuntimeError(f"unsupported shape for in_queue rebatching. expect 1 or 2. while {to_batch[0].shape}")
        elif isinstance(to_batch[0], list):
            batched[key] = []
            for seq in to_batch:
                batched[key].extend(seq)
        else:
            raise Exception(f"unknown types key: {key} and {type(to_batch[0])} to concat : {to_batch[0]}")

    return batched

def slice_by_index_along_batch(batched_input: Dict[str, Union[torch.Tensor, List[Any]]], index):
    start = index[0]
    offset = index[1]
    batched = {}
    for key in batched_input.keys():
        if isinstance(batched_input[key], torch.Tensor):
            batched[key] = batched_input[key][start::offset,...]
        elif isinstance(batched_input[key], list):
            batched[key] = batched_input[key][start::offset]
    return batched

def slice_data_list_by_index(batched_input: List[Dict[str, Any]], index):
    """
    Slice input data_list by slice index
    """
    total_length = len(batched_input)
    slice_id = index[0]
    total_slice = index[1]
    slice_size = total_length // total_slice
    # When total_length % total_slice != 0, Remaining data will be append to last slice to avoid droping data
    reminder = total_length % total_slice
    start_index = slice_id * slice_size
    end_index = start_index + slice_size if slice_id != total_slice - 1 else start_index + slice_size + reminder
    return batched_input[start_index : end_index]

def listdict_to_dictlist(ld, list_extend=True):
    '''
    [{k1: v11, k2: v2}, {k1: v12, k2: v2},....] => {k1: [v11, v12..], k2: [v21, v22...]}
    if v11 is list then k1: v11 + v12
    :param ld:
    :return:
    '''
    res = copy.deepcopy(ld[0])
    for res_key, v in res.items():
        if list_extend and isinstance(res[res_key], list):
            continue

        res[res_key] = [v]

    for d in ld[1:]:
        for key, v in d.items():
            if list_extend and isinstance(d[key], list):
                res[key].extend(v)
            else:
                res[key].append(v)

    return res


def map_metrics(metric_list):
    mapped_metrics = {}
    for metrics in metric_list:
        for key, value in metrics.items():
            if key in mapped_metrics:
                mapped_metrics[key].append(value)
            else:
                mapped_metrics[key] = [value]
    return mapped_metrics


def reduce_metrics(merged_metrics):
    # [TODO:baodong.lh] support custom_op like min, max to reduce metrics
    reduced_metrics = {}
    for key, value_list in merged_metrics.items():
        if isinstance(value_list[0], torch.Tensor):
            value = torch.mean(torch.Tensor(value_list))
        else:
            value = np.mean(value_list)
        reduced_metrics[key] = value
    return reduced_metrics


def map_reduce_metrics(metric_list):
    # [TODO:baodong.lh] imporve performance by distributing the task to per-replica
    # sanity check
    assert isinstance(metric_list, list)

    if len(metric_list) == 0:
        return {}

    first_metric_len = len(metric_list[0])
    for i, metric in enumerate(metric_list):
        if len(metric) != first_metric_len:
            logger.info(
                f"WARNING! length of metrics are not the same for {i}-th metric ({len(metric)}) "
                f"and the first one ({first_metric_len})! This is weird and please check!"
            )

    mapped_metrics = map_metrics(metric_list)
    reduced_metrics = reduce_metrics(mapped_metrics)
    return reduced_metrics

# Copied from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/utils.py
def get_te_version():
    """Get TE version from __version__; if not available use pip's. Use caching."""
    if not HAVE_PACKAGING:
        raise ImportError(
            "packaging is not installed. Please install it with `pip install packaging`."
        )

    try:
        import transformer_engine as te # pylint: disable=import-outside-toplevel, unused-import

        HAVE_TE = True
    except ImportError:
        HAVE_TE = False

    def get_te_version_str():
        import transformer_engine as te # pylint: disable=import-outside-toplevel

        if hasattr(te, "__version__"):
            return str(te.__version__)
        else:
            return version("transformer-engine")

    global _te_version
    if _te_version is None and HAVE_TE:
        _te_version = PkgVersion(get_te_version_str())
    return _te_version


def is_te_min_version(version, check_equality=True):
    """Check if minimum version of `transformer-engine` is installed."""
    if not HAVE_PACKAGING:
        raise ImportError(
            "packaging is not installed. Please install it with `pip install packaging`."
        )

    if check_equality:
        return get_te_version() >= PkgVersion(version)
    return get_te_version() > PkgVersion(version)
