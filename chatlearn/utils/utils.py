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
"""utils"""

import ast
import inspect
import socket
import subprocess
import textwrap
import time
import math
import concurrent.futures
from contextlib import closing
from types import SimpleNamespace

import pynvml
import torch
from chatlearn.utils.logger import logger


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


def get_or_cache(cache, key, func):
    """
    get results if cached
    otherwise call the func to get the results, and cache the results
    """
    if key in cache:
        return cache[key]
    res = func()
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


def execute_in_parallel(function, arguments):
    if len(arguments) == 1:
        return function(*arguments[0])
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Using list comprehension to handle the results
        futures = [executor.submit(function, *args) for args in arguments]
        for _future in concurrent.futures.as_completed(futures):
            results.append(_future.result())
    return results


def multi_thread_data_processing(num_threads: int, all_data: list, process_one_data, fn_args: list):
    num_data = len(all_data)

    if num_data == 0:
        return []
    # reduce num_threads if data amount is little
    if num_data < num_threads:
        num_threads = num_data
    assert num_threads > 0, "Get num_threads <= 0. Expect to be a positive number."

    result = list(range(num_data))
    data_size_per_thread = math.ceil(num_data / num_threads)
    thread_args = [(i, all_data[i * data_size_per_thread : min((i+1) * data_size_per_thread, num_data)]) for i in range(num_threads)]

    def thread_fn(thread_id: int, pending_data):
        offset = thread_id * data_size_per_thread
        for i, data in enumerate(pending_data):
            result[offset + i] = process_one_data(data, *fn_args)

    execute_in_parallel(thread_fn, thread_args)

    return result
