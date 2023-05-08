import ast
import inspect
import socket
import textwrap
from contextlib import closing

import ray
import torch
from tqdm import tqdm


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
    res = node_iter.visit(ast.parse(code))
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


def get(data):
    if isinstance(data, (list, tuple)):
        dtype = type(data)
        ret = dtype(get(item) for item in data)
        return ret
    if isinstance(data, dict):
        return {key: get(value) for key, value in data.items()}
    while isinstance(data, ray.ObjectRef):
        data = ray.get(data)
    return data


def split_index(length, num_splits):
    # Calculate the size of each split
    size = length // num_splits
    remainder = length % num_splits

    # Initialize an empty list for indices
    indices = []

    # Loop over the number of splits and append indices
    start = 0
    end = 0
    for i in range(num_splits):
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


def wait(refs, desc=None):
    """
    wait until all computation finish
    TODO: note this function will hide errors!
    """
    if desc is not None:
        pbar = tqdm(total=len(refs), desc=desc)
    while refs:
        done, refs = ray.wait(refs)
        if desc is not None:
            pbar.update(len(done))
    if desc is not None:
        pbar.close()


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
