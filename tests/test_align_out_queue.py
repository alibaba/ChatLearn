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
"""UT for out_queue alignment."""

import torch
from ray.util.queue import Queue
from chatlearn.runtime.executor import Executor


def test_align():
    queues = []
    for j in [1,2,4]:
        num_producers = 2 * j
        in_queue = Queue()
        for i in range(num_producers):
            item = {
                f"tensor_{j}": torch.rand(8//num_producers,4+i),
                f"list_{j}": [[1+k+i,2+k+i] for k in range(8//num_producers)]
            }
            in_queue.put(item)
        queues.append(in_queue)

    assert [ele.qsize() for ele in queues] == [2, 4, 8]

    out_queues = Executor.align_out_queues(queues, encode=False)

    assert [ele.qsize() for ele in out_queues] == [2, 2, 2]

    for out_queue in out_queues:
        kvs = out_queue.get()
        for key, value in kvs.items():
            if "tensor" in key:
                assert value.shape[0] == 4
            else:
                assert len(value) == 4


if __name__ == '__main__':
    test_align()
