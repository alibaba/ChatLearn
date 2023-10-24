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
"""
Tools to convert megatron checkpoint with different parallel strategies.
Add support for reward model conversion.
"""
# pylint: disable=wildcard-import,exec-used
import sys
import importlib

def get_indent_count(string):
    count = 0
    for s in string:
        if s == ' ':
            count += 1
        else:
            return count

def repair_util(source):
    source = source.replace("choices=['GPT', 'BERT']", "choices=['GPT', 'BERT', 'REWARD']")
    return source


def detect_and_insert_code(lines, pattern, new_code, additional_indent, line_offset):
    type_line_number, type_line = [(line_number, line) for line_number, line in enumerate(lines) if pattern in line][0]
    indent = get_indent_count(type_line) + additional_indent
    new_lines = [line for line in new_code.split('\n') if line.strip()]
    added_lines = []
    for line in new_lines:
        added_lines.append(" "*indent + line)
    lines = lines[:type_line_number+line_offset] + added_lines + lines[type_line_number+line_offset:]
    return lines

def repair_loader_model_provider(lines):
    # Insert before following code, so line_offset=-2
    # else:
    #     raise Exception(f'unrecognized model type: {args.model_type}')
    pattern = 'unrecognized model type'
    new_code = \
"""
elif args.model_type == 'REWARD':
    from models.reward_model import model_provider
    margs.model_type = ModelType.encoder_or_decoder
"""
    indent = -4
    line_offset = -2
    return detect_and_insert_code(lines, pattern, new_code, indent, line_offset)

def repair_saver_model_provider(lines):
    return repair_loader_model_provider(lines)

def repair_loader_put_reward(lines):
    pattern = 'queue.put("done")'
    new_code = \
"""
if md.model_type == 'REWARD':
    print("Sending LM Pooler")
    message = {
        "weight1": models[0].pooler_head.dense1.weight.data,
        "bias1": models[0].pooler_head.dense1.bias.data,
        "weight2": models[0].pooler_head.dense2.weight.data,
        "bias2": models[0].pooler_head.dense2.bias.data,
    }
    queue_put("pooler_head", message)
"""
    return detect_and_insert_code(lines, pattern, new_code, 0, 0)

def repair_saver_put_reward(lines):
    pattern = 'if msg != "done":'
    new_code = \
"""
if msg != "done" and msg["name"] == "pooler_head":
    if not hasattr(models[0], 'pooler_head'):
        print("ERROR: got a pooler_head, but model does not have one")
        exit(1)
    print("received pooler_head")
    head_weight1 = msg.pop("weight1")
    head_bias1 = msg.pop("bias1")
    head_weight2 = msg.pop("weight2")
    head_bias2 = msg.pop("bias2")
    for tp_rank in range(args.target_tensor_parallel_size):
        models[tp_rank].pooler_head.dense1.weight.data.copy_(head_weight1)
        models[tp_rank].pooler_head.dense1.bias.data.copy_(head_bias1)
        models[tp_rank].pooler_head.dense2.weight.data.copy_(head_weight2)
        models[tp_rank].pooler_head.dense2.bias.data.copy_(head_bias2)
    check_message(msg)
    msg = queue_get()
"""
    return detect_and_insert_code(lines, pattern, new_code, 0, 0)


class CheckpointUtilsImporter:
    """CheckpointUtilsImporter"""

    def __init__(self, *args):
        self.module_names = args
        self.path = None

    def find_module(self, fullname, path=None):
        if fullname in self.module_names:
            # save the path so that it could be used later by `load_module`
            self.path = path
            return self
        return None

    def repair_code(self, source, module_name):

        if module_name == 'util':
            source = repair_util(source)
        elif module_name == 'loader_megatron':
            lines = source.split('\n')
            lines = repair_loader_model_provider(lines)
            lines = repair_loader_put_reward(lines)
            source = '\n'.join(lines)
        elif module_name == 'saver_megatron':
            lines = source.split('\n')
            lines = repair_saver_model_provider(lines)
            lines = repair_saver_put_reward(lines)
            source = '\n'.join(lines)
        return source

    def load_module(self, name):
        """
        Load the module source code, fix the source and import manually
        :param name:
        :return:
        """
        if name in sys.modules:
            return sys.modules[name]
        module_name = name.split('.')[-1]
        module_path = self.path[0] + '/' + module_name + '.py'

        # create the module spec object
        spec = importlib.util.spec_from_file_location(name, module_path)

        # read the source code and modify on-the-fly
        with open(module_path, encoding="utf-8") as f:
            source = f.read()
        new_source = self.repair_code(source, module_name)

        # create the module object based off the module spec
        module = importlib.util.module_from_spec(spec)

        # compile the source code into a code object where it
        # could be imported with `exec` call.
        codeobj = compile(new_source, module.__spec__.origin, 'exec')

        # module.__dict__ is required for referencing variables in the module
        exec(codeobj, module.__dict__)  # pylint: disable=exec-used

        # put the loaded module into sys.modules so that if the module is imported
        # again it could be found.
        sys.modules[name] = module
        if 'loader_megatron' in name or 'saver_megatron' in name:
            sys.modules[module_name] = module

        # return the module itself so that it could be used
        return module

if __name__ == '__main__':
    sys.meta_path.insert(-1, CheckpointUtilsImporter('tools.checkpoint.util', 'tools.checkpoint.loader_megatron', 'tools.checkpoint.saver_megatron'))
    from tools.checkpoint import loader_megatron, saver_megatron # pylint: disable=unused-import
    from tools.checkpoint import util
    util.main()
# pylint: enable=wildcard-import,exec-used
