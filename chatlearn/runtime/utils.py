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
import textwrap
import inspect
from collections import defaultdict

def encode_data(mb, data):
    return {"iter": mb, "data": data}


def decode_data(data):
    mb = data["iter"]
    data = data["data"]
    return mb, data


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
        self.model_to_call_funcs = defaultdict(list)

    def parse_assign(self, line):
        func_name, model_name, _ = parse_expr(line)
        model = self.global_models[model_name]
        self.model_to_call_funcs[model].append(func_name)

    def visit_func(self, node):
        for line in node.body:
            if isinstance(line, (ast.Assign, ast.Expr)):
                self.parse_assign(line)
            elif isinstance(line, ast.With):
                for line0 in line.body:
                    if isinstance(line0, (ast.Assign, ast.Expr)):
                        self.parse_assign(line0)

    def parse(self, func):
        closure_vars = inspect.getclosurevars(func)
        self.global_models = {}
        if closure_vars.globals:
            self.global_models.update(closure_vars.globals)
        if closure_vars.nonlocals:
            self.global_models.update(closure_vars.nonlocals)
        node_iter = ast.NodeVisitor()
        node_iter.visit_FunctionDef = self.visit_func
        if isinstance(func, str):
            code = textwrap.dedent(func)
        else:
            code = textwrap.dedent(inspect.getsource(func))
        node_iter.visit(ast.parse(code))
        return self.model_to_call_funcs
