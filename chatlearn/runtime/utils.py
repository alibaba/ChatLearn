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
"""runtime utils"""

import ast
import concurrent.futures
import textwrap
import inspect
from chatlearn.utils import future


def encode_data(mb, data):
    return {"iter": mb, "data": data}


def decode_data(data):
    mb = data["iter"]
    data = data["data"]
    return mb, data


def build_scheduler(model_replica):
    # build for only last rank of each replica.
    future.get(model_replica.tailer.build_scheduler.remote())


def free_cache_engine(model_replica):
    rets = []
    for actor in model_replica.all_actors:
        rets.append(actor.free_cache_engine.remote())
    rets = future.get(rets)


def prepare_vllm(model_replica):
    """Profiling cache blocks and build scheduler."""
    profile_cache_blocks(model_replica)
    # setup vllm scheduler
    build_scheduler(model_replica)


def profile_cache_blocks(model_replica):
    rets = []
    for actor in model_replica.all_actors:
        rets.append(actor.profile_cache_blocks.remote())
    rets = future.get(rets)

    num_gpu_blocks = min(a[0] for a in rets)
    num_cpu_blocks = min(a[1] for a in rets)

    rets = []
    for actor in model_replica.all_actors:
        rets.append(actor.set_cache_config.remote(num_gpu_blocks, num_cpu_blocks))
    rets = future.get(rets)


def reinit_cache_engine(model_replica):
    rets = []
    for actor in model_replica.all_actors:
        rets.append(actor.reinit_cache_engine.remote())
    rets = future.get(rets)


def vllm_post_process_generate_step_one_model(replica, out_queue, mb):
    """
    Args:
        model: DistModel
        out_queue: Queue
    """
    output = replica.tailer.decode.remote()

    free_cache_engine(replica)

    # If tp > 1 or pp > 1 for current model, its `output` will be a list whose
    #   length is the number of Actors. In this case, all members in the list
    #   are the same, and we choose output[-1] to put into out_queue.
    last_output = output[-1] if isinstance(output, list) else output
    if isinstance(out_queue, list):
        for oq in out_queue:
            oq.put(encode_data(mb, last_output))
    else:
        out_queue.put(encode_data(mb, last_output))


def parse_assign_target(line):
    targets = []
    for target in line.targets:
        targets.append(target.id)
    return targets


def parse_expr(line):
    func = line.value.func
    func_name = func.attr
    func_args = [arg.id for arg in line.value.args]
    if isinstance(func.value, ast.Name):
        model_name = func.value.id
    else:
        model_name = func.value.attr
    return func_name, model_name, func_args


class FlowParser:
    """Flow Parser"""

    def __init__(self):
        self.model_to_call_func = {}

    def visit_func(self, node):
        for line in node.body:
            if isinstance(line, (ast.Assign, ast.Expr)):
                func_name, model_name, _ = parse_expr(line)
                model = self.global_models[model_name]
                assert model not in self.model_to_call_func
                self.model_to_call_func[model] = func_name

    def parse(self, func):
        closure_vars = inspect.getclosurevars(func)
        self.global_models = closure_vars.globals if closure_vars.globals else closure_vars.nonlocals
        node_iter = ast.NodeVisitor()
        node_iter.visit_FunctionDef = self.visit_func
        if isinstance(func, str):
            code = textwrap.dedent(func)
        else:
            code = textwrap.dedent(inspect.getsource(func))
        node_iter.visit(ast.parse(code))
        return self.model_to_call_func

def execute_in_parallel(function, arguments):
    if len(arguments) == 1:
        return function(*arguments[0])
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Using list comprehension to handle the results
        futures = [executor.submit(function, *args) for args in arguments]
        for _future in concurrent.futures.as_completed(futures):
            try:
                results.append(_future.result())
            except Exception as e:
                print(f"Thread generated an exception: {e}")
    return results
