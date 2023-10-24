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
Add attention_acc kernel to speedup Attention when seq_len=1
"""

# pylint: disable=unused-import,wildcard-import
import inspect
import torch
try:
    from megatron import get_args
    from megatron.core import mpu, tensor_parallel
    from megatron.model.enums import AttnType
    from megatron.model.transformer import ParallelAttention
    from megatron.model.transformer import *
except ImportError:
    ParallelAttention = None
try:
    from einops import rearrange
except ImportError:
    rearrange = None
# pylint: enable=unused-import,wildcard-import
from chatlearn.utils.utils import detect_and_insert_code_to_func


def add_attn_acc_one_seq_kernel(source_code):
    if 'elif not self.use_flash_attn:' in source_code:
        return
    pattern = 'if not self.use_flash_attn:'
    new_code = \
"""
args = get_args()
use_attn_acc = hasattr(args, 'use_attn_acc') and args.use_attn_acc
if use_attn_acc and query_layer.size(0) == 1:
    import attention_acc
    context_layer = attention_acc.mha(
        query_layer,
        key_layer,
        value_layer
    )
    context_layer = context_layer.view(context_layer.size(0), context_layer.size(1), -1)
"""
    source_code = detect_and_insert_code_to_func(source_code, pattern, new_code)
    if source_code is None:
        return
    source_code = source_code.replace('if not self.use_flash_attn:', 'elif not self.use_flash_attn:')
    return source_code

def apply_rotary_pos_emb_variable_seq(source_code):
    pattern = 'rotary_pos_emb = (q_pos_emb, k_pos_emb)'
    new_code = \
"""
else:
    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb
        sequence_end = query_layer.size(0)
        q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
        k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
        rotary_pos_emb = (q_pos_emb, k_pos_emb)
"""
    return detect_and_insert_code_to_func(source_code, pattern, new_code, -8, 1)

def modify_code(source):
    source = add_attn_acc_one_seq_kernel(source)
    if source is not None:
        return apply_rotary_pos_emb_variable_seq(source)
    return source

if ParallelAttention is not None:
    src_code = inspect.getsource(ParallelAttention.forward)
    if 'use_attn_acc' not in src_code:
        src_code = modify_code(src_code)
        if src_code is not None:
            exec(src_code) # pylint: disable=exec-used
            ParallelAttention.forward = forward
