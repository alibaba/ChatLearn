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
from typing import Tuple, List, Any

import chatlearn
from chatlearn.utils.mappings import ShardedTensorInfo
from chatlearn.synchronizer.mappers.mapping_helpers import (
    process_normal_tensor,
    process_gate_up_tensor,
    process_qkv_tensor
)

def build_answer(
    global_shape: Tuple[int, ...],
    squeezed_answer: List[Any]
):
    # axis_fragmentations, global_offset, local_shape, local_offset
    maybe_wrap_tuple = lambda x: (x, ) if isinstance(x, int) else x
    answer = []
    for src, tgt in squeezed_answer:
        answer.append((
            ShardedTensorInfo(
                axis_fragmentations=maybe_wrap_tuple(src[0]), 
                global_shape=global_shape, 
                global_offset=maybe_wrap_tuple(src[1]), 
                local_shape=maybe_wrap_tuple(src[2]), 
                local_offset=maybe_wrap_tuple(src[3])
            ),
            ShardedTensorInfo(
                axis_fragmentations=maybe_wrap_tuple(tgt[0]), 
                global_shape=global_shape, 
                global_offset=maybe_wrap_tuple(tgt[1]), 
                local_shape=maybe_wrap_tuple(tgt[2]), 
                local_offset=maybe_wrap_tuple(tgt[3])
            )
        ))
    return answer

def test_process_normal_tensor():
    # Case 1: src tp == dst tp
    assert process_normal_tensor(
        ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(16, 8), global_offset=(0, 0)),
        4
    ) == [
        (
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(16, 8), global_offset=(0, 0)), 
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(16, 8), global_offset=(0, 0))
        )
    ], "Case 1: src tp == dst tp failed"

    # Case 2: src tp < dst tp
    assert sorted(process_normal_tensor(
        ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(16, ), global_offset=(1, )),
        4
    ), key=lambda x: x[0].local_offset[0]) == [
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(16, ), global_offset=(1, ), local_shape=(4, ), local_offset=(0, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(16, ), global_offset=(2, ))
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(16, ), global_offset=(1, ), local_shape=(4, ), local_offset=(4, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(16, ), global_offset=(3, ))
        )
    ], "Case 2: src tp < dst tp failed"

    # Case 3: src tp > dst tp
    assert sorted(process_normal_tensor(
        ShardedTensorInfo(axis_fragmentations=(1, 8), global_shape=(3, 16), global_offset=(0, 3)),
        2, 
        axis=1
    ), key=lambda x: x[0].local_offset[1]) == [
        (
            ShardedTensorInfo(axis_fragmentations=(1, 8), global_shape=(3, 16), global_offset=(0, 3)), 
            ShardedTensorInfo(axis_fragmentations=(1, 2), global_shape=(3, 16), global_offset=(0, 0), local_shape=(3, 2), local_offset=(0, 6))
        )
    ], "Case 3: src tp > dst tp failed"

    # Case 4: src tp = 3 and dst tp = 2
    assert sorted(process_normal_tensor(
        ShardedTensorInfo(axis_fragmentations=(3, 2, 1), global_shape=(6, 2, 3), global_offset=(1, 1, 0)),
        2, 
    ), key=lambda x: x[0].local_offset[0]) == [
        (
            ShardedTensorInfo(axis_fragmentations=(3, 2, 1), global_shape=(6, 2, 3), global_offset=(1, 1, 0), local_offset=(0, 0, 0), local_shape=(1, 1, 3)),
            ShardedTensorInfo(axis_fragmentations=(2, 2, 1), global_shape=(6, 2, 3), global_offset=(0, 1, 0), local_offset=(2, 0, 0), local_shape=(1, 1, 3)),
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, 2, 1), global_shape=(6, 2, 3), global_offset=(1, 1, 0), local_offset=(1, 0, 0), local_shape=(1, 1, 3)),
            ShardedTensorInfo(axis_fragmentations=(2, 2, 1), global_shape=(6, 2, 3), global_offset=(1, 1, 0), local_offset=(0, 0, 0), local_shape=(1, 1, 3)),
        ),
    ], "Case 4: src tp = 3 and dst tp = 2 failed"

    # Case 5: src tp = 3 and dst tp = 8
    assert sorted(process_normal_tensor(
        ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(24, ), global_offset=(1, )), # 8 ~ 15
        8, 
    ), key=lambda x: x[0].local_offset[0]) == [
        (
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(24, ), global_offset=(1, ), local_offset=(0, ), local_shape=(1, )),
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(24, ), global_offset=(2, ), local_offset=(2, ), local_shape=(1, )),
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(24, ), global_offset=(1, ), local_offset=(1, ), local_shape=(3, )),
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(24, ), global_offset=(3, ),),
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(24, ), global_offset=(1, ), local_offset=(4, ), local_shape=(3, )),
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(24, ), global_offset=(4, ),),
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(24, ), global_offset=(1, ), local_offset=(7, ), local_shape=(1, )),
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(24, ), global_offset=(5, ), local_offset=(0, ), local_shape=(1, )),
        ),
    ], "Case 5: src tp = 3 and dst tp = 8 failed"

def test_process_gate_up_tensor():
    # Case 1: src tp == dst tp
    assert process_gate_up_tensor(
        ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(16, 8), global_offset=(0, 0)),
        4
    ) == [
        (
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(16, 8), global_offset=(0, 0)), 
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(16, 8), global_offset=(0, 0))
        )
    ], "Case 1: src tp == dst tp failed"

    # Case 2: src tp < dst tp
    assert sorted(process_gate_up_tensor(
        ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(16, ), global_offset=(1, )),
        4
    ), key=lambda x: x[0].local_offset[0]) == [
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(16, ), global_offset=(1, ), local_shape=(2, ), local_offset=(0, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(16, ), global_offset=(2, ), local_shape=(2, ), local_offset=(0, ))
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(16, ), global_offset=(1, ), local_shape=(2, ), local_offset=(2, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(16, ), global_offset=(3, ), local_shape=(2, ), local_offset=(0, ))
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(16, ), global_offset=(1, ), local_shape=(2, ), local_offset=(4, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(16, ), global_offset=(2, ), local_shape=(2, ), local_offset=(2, ))
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(16, ), global_offset=(1, ), local_shape=(2, ), local_offset=(6, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(16, ), global_offset=(3, ), local_shape=(2, ), local_offset=(2, ))
        ),
    ], "Case 2: src tp < dst tp failed"

    # Case 3: src tp > dst tp
    assert sorted(process_gate_up_tensor(
        ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(16, ), global_offset=(3, )),
        2, 
    ), key=lambda x: x[0].local_offset[0]) == [
        (
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(16, ), global_offset=(3, ), local_shape=(1, ), local_offset=(0, )), 
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(16, ), global_offset=(0, ), local_shape=(1, ), local_offset=(3, ))
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(16, ), global_offset=(3, ), local_shape=(1, ), local_offset=(1, )), 
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(16, ), global_offset=(0, ), local_shape=(1, ), local_offset=(7, ))
        ),    
    ], "Case 3: src tp > dst tp failed"

    # Case 4: src tp = 3 and dst tp = 2
    assert sorted(process_gate_up_tensor(
        ShardedTensorInfo(axis_fragmentations=(3, 1), global_shape=(24, 7), global_offset=(1, 0)),
        2, 
    ), key=lambda x: x[0].local_offset[0]) == [
        (
            ShardedTensorInfo(axis_fragmentations=(3, 1), global_shape=(24, 7), global_offset=(1, 0), local_shape=(2, 7), local_offset=(0, 0)),
            ShardedTensorInfo(axis_fragmentations=(2, 1), global_shape=(24, 7), global_offset=(0, 0), local_shape=(2, 7), local_offset=(4, 0))
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, 1), global_shape=(24, 7), global_offset=(1, 0), local_shape=(2, 7), local_offset=(2, 0)),
            ShardedTensorInfo(axis_fragmentations=(2, 1), global_shape=(24, 7), global_offset=(1, 0), local_shape=(2, 7), local_offset=(0, 0))
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, 1), global_shape=(24, 7), global_offset=(1, 0), local_shape=(2, 7), local_offset=(4, 0)),
            ShardedTensorInfo(axis_fragmentations=(2, 1), global_shape=(24, 7), global_offset=(0, 0), local_shape=(2, 7), local_offset=(10, 0))
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, 1), global_shape=(24, 7), global_offset=(1, 0), local_shape=(2, 7), local_offset=(6, 0)),
            ShardedTensorInfo(axis_fragmentations=(2, 1), global_shape=(24, 7), global_offset=(1, 0), local_shape=(2, 7), local_offset=(6, 0))
        ),
    ], "Case 4: src tp = 3 and dst tp = 2 failed"

    # Case 5: src tp = 3 and dst tp = 8
    assert sorted(process_gate_up_tensor(
        ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(48, ), global_offset=(1, )),
        8, 
    ), key=lambda x: x[0].local_offset[0]) == [
        (
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(48, ), global_offset=(1, ), local_offset=(0, ), local_shape=(1, )),
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(48, ), global_offset=(2, ), local_offset=(2, ), local_shape=(1, )),
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(48, ), global_offset=(1, ), local_offset=(1, ), local_shape=(3, )),
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(48, ), global_offset=(3, ), local_offset=(0, ), local_shape=(3, )),
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(48, ), global_offset=(1, ), local_offset=(4, ), local_shape=(3, )),
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(48, ), global_offset=(4, ), local_offset=(0, ), local_shape=(3, )),
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(48, ), global_offset=(1, ), local_offset=(7, ), local_shape=(1, )),
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(48, ), global_offset=(5, ), local_offset=(0, ), local_shape=(1, )),
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(48, ), global_offset=(1, ), local_offset=(8, ), local_shape=(1, )),
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(48, ), global_offset=(2, ), local_offset=(5, ), local_shape=(1, )),
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(48, ), global_offset=(1, ), local_offset=(9, ), local_shape=(3, )),
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(48, ), global_offset=(3, ), local_offset=(3, ), local_shape=(3, )),
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(48, ), global_offset=(1, ), local_offset=(12, ), local_shape=(3, )),
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(48, ), global_offset=(4, ), local_offset=(3, ), local_shape=(3, )),
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(48, ), global_offset=(1, ), local_offset=(15, ), local_shape=(1, )),
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(48, ), global_offset=(5, ), local_offset=(3, ), local_shape=(1, )),
        ),
    ], "Case 5: src tp = 3 and dst tp = 8 failed"
def test_process_qkv_tensor_no_gqa():
    # Case 1: src tp == dst tp
    assert process_qkv_tensor(
        ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0)),
        8,
        None,
        4
    ) == [
        (
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0), local_shape=(4, 8), local_offset=(0, 0)), 
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0), local_shape=(4, 8), local_offset=(0, 0)), 
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0), local_shape=(4, 8), local_offset=(4, 0)), 
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0), local_shape=(4, 8), local_offset=(8, 0)), 
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0), local_shape=(4, 8), local_offset=(8, 0)), 
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0), local_shape=(4, 8), local_offset=(16, 0)), 
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0), local_shape=(4, 8), local_offset=(12, 0)), 
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0), local_shape=(4, 8), local_offset=(4, 0)), 
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0), local_shape=(4, 8), local_offset=(16, 0)), 
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0), local_shape=(4, 8), local_offset=(12, 0)), 
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0), local_shape=(4, 8), local_offset=(20, 0)), 
            ShardedTensorInfo(axis_fragmentations=(4, 1), global_shape=(96, 8), global_offset=(0, 0), local_shape=(4, 8), local_offset=(20, 0)), 
        ),
    ], "Case 1: src tp == dst tp failed"

    # Case 2: src tp < dst tp
    assert sorted(process_qkv_tensor(
        ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, )),
        8,
        None,
        4
    ), key=lambda x: x[0].local_offset[0]) == [
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, ), local_shape=(4, ), local_offset=(0, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(96, ), global_offset=(2, ), local_shape=(4, ), local_offset=(0, )), 
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, ), local_shape=(4, ), local_offset=(4, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(96, ), global_offset=(2, ), local_shape=(4, ), local_offset=(8, )), 
        ), 
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, ), local_shape=(4, ), local_offset=(8, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(96, ), global_offset=(2, ), local_shape=(4, ), local_offset=(16, )), 
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, ), local_shape=(4, ), local_offset=(12, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(96, ), global_offset=(2, ), local_shape=(4, ), local_offset=(4, )), 
        ), 
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, ), local_shape=(4, ), local_offset=(16, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(96, ), global_offset=(2, ), local_shape=(4, ), local_offset=(12, )), 
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, ), local_shape=(4, ), local_offset=(20, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(96, ), global_offset=(2, ), local_shape=(4, ), local_offset=(20, )), 
        ), 
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, ), local_shape=(4, ), local_offset=(24, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(96, ), global_offset=(3, ), local_shape=(4, ), local_offset=(0, )), 
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, ), local_shape=(4, ), local_offset=(28, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(96, ), global_offset=(3, ), local_shape=(4, ), local_offset=(8, )), 
        ), 
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, ), local_shape=(4, ), local_offset=(32, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(96, ), global_offset=(3, ), local_shape=(4, ), local_offset=(16, )), 
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, ), local_shape=(4, ), local_offset=(36, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(96, ), global_offset=(3, ), local_shape=(4, ), local_offset=(4, )), 
        ), 
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, ), local_shape=(4, ), local_offset=(40, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(96, ), global_offset=(3, ), local_shape=(4, ), local_offset=(12, )), 
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(96, ), global_offset=(1, ), local_shape=(4, ), local_offset=(44, )), 
            ShardedTensorInfo(axis_fragmentations=(4, ), global_shape=(96, ), global_offset=(3, ), local_shape=(4, ), local_offset=(20, )), 
        ), 
    ], "Case 2: src tp < dst tp failed"

    # Case 3: src tp > dst tp
    assert sorted(process_qkv_tensor(
        ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(48, ), global_offset=(3, )),
        8,
        None,
        2, 
    ), key=lambda x: x[0].local_offset[0]) == [
        (
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(48, ), global_offset=(3, ), local_shape=(2, ), local_offset=(0, )), 
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(48, ), global_offset=(0, ), local_shape=(2, ), local_offset=(6, ))
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(48, ), global_offset=(3, ), local_shape=(2, ), local_offset=(2, )), 
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(48, ), global_offset=(0, ), local_shape=(2, ), local_offset=(14, ))
        ),
        (
            ShardedTensorInfo(axis_fragmentations=(8, ), global_shape=(48, ), global_offset=(3, ), local_shape=(2, ), local_offset=(4, )), 
            ShardedTensorInfo(axis_fragmentations=(2, ), global_shape=(48, ), global_offset=(0, ), local_shape=(2, ), local_offset=(22, ))
        ),  
    ], "Case 3: src tp > dst tp failed"

    # Case 4: src tp = 3 and dst tp = 2
    calculated = sorted(
        process_qkv_tensor(
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(72, ), global_offset=(1, )),
            12,
            None,
            2, 
        ),
        key=lambda x: x[0].local_offset[0]
    )
    # axis_fragmentations, global_offset, local_shape, local_offset
    expected = [
        [[3, 1, 2, 0], [2, 0, 2, 8]],
        [[3, 1, 2, 2], [2, 0, 2, 20]],
        [[3, 1, 2, 4], [2, 0, 2, 32]],
        [[3, 1, 2, 6], [2, 0, 2, 10]],
        [[3, 1, 2, 8], [2, 0, 2, 22]],
        [[3, 1, 2, 10], [2, 0, 2, 34]],
        [[3, 1, 2, 12], [2, 1, 2, 0]],
        [[3, 1, 2, 14], [2, 1, 2, 12]],
        [[3, 1, 2, 16], [2, 1, 2, 24]],
        [[3, 1, 2, 18], [2, 1, 2, 2]],
        [[3, 1, 2, 20], [2, 1, 2, 14]],
        [[3, 1, 2, 22], [2, 1, 2, 26]],
    ]
    assert calculated == build_answer((72, ), expected), "Case 4: src tp = 3 and dst tp = 2 failed"

    # Case 5: src tp = 3 and dst tp = 4
    calculated = sorted(
        process_qkv_tensor(
            ShardedTensorInfo(axis_fragmentations=(3, ), global_shape=(36, ), global_offset=(1, )),
            12,
            None,
            4, 
        ), 
        key=lambda x: x[0].local_offset[0]
    )
    # axis_fragmentations, global_offset, local_shape, local_offset
    expected = [
        [[3, 1, 1, 0], [4, 1, 1, 1]],
        [[3, 1, 1, 1], [4, 1, 1, 4]],
        [[3, 1, 1, 2], [4, 1, 1, 7]],
        [[3, 1, 1, 3], [4, 1, 1, 2]],
        [[3, 1, 1, 4], [4, 1, 1, 5]],
        [[3, 1, 1, 5], [4, 1, 1, 8]],
        [[3, 1, 1, 6], [4, 2, 1, 0]],
        [[3, 1, 1, 7], [4, 2, 1, 3]],
        [[3, 1, 1, 8], [4, 2, 1, 6]],
        [[3, 1, 1, 9], [4, 2, 1, 1]],
        [[3, 1, 1, 10], [4, 2, 1, 4]],
        [[3, 1, 1, 11], [4, 2, 1, 7]],
    ]
    assert calculated == build_answer((36, ), expected), "Case 5: src tp = 3 and dst tp = 4 failed"

TEST_CASE = [
    test_process_normal_tensor,
    test_process_gate_up_tensor,
    test_process_qkv_tensor_no_gqa
]

if __name__ == "__main__":
    for case in TEST_CASE:
        case()