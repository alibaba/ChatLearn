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
"""
Add attention_acc kernel to speedup Attention when seq_len=1
"""

# pylint: disable=unused-import,wildcard-import,unused-wildcard-import,ungrouped-imports
import inspect

try:
    from chatlearn.utils.megatron_import_transformer_helper import ParallelAttention
    from chatlearn.utils.megatron_import_transformer_helper import *
except ImportError:
    ParallelAttention = None


from chatlearn.utils.utils import detect_and_insert_code_to_func
# pylint: enable=unused-import,wildcard-import,unused-wildcard-import,ungrouped-imports


def apply_rotary_pos_emb_variable_seq(source_code):
    pattern = 'rotary_pos_emb = (q_pos_emb, k_pos_emb)'
    new_code = \
"""
else:
    # apply_rotary_pos_emb_variable_seq
    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb
        sequence_end = query_layer.size(0)
        q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
        k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
        rotary_pos_emb = (q_pos_emb, k_pos_emb)
"""
    return detect_and_insert_code_to_func(source_code, pattern, new_code, -8, 1)

def modify_code(source):
    if source is not None:
        return apply_rotary_pos_emb_variable_seq(source)

if ParallelAttention is not None:
    src_code = inspect.getsource(ParallelAttention.forward)
    if '# apply_rotary_pos_emb_variable_seq' not in src_code:
        src_code = modify_code(src_code)
        if src_code is not None:
            exec(src_code) # pylint: disable=exec-used
            ParallelAttention.forward = forward
