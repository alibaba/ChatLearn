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
"""Check ParameterSync"""

import traceback
import argparse
import os
import concurrent
from concurrent.futures import ThreadPoolExecutor
import torch
def check_tensor_value(expected_fname, synced_fname):
    expected_tensor = torch.load(expected_fname, map_location="cpu")
    synced_tensor = torch.load(synced_fname, map_location="cpu")
    if not torch.allclose(expected_tensor, synced_tensor):
        return f"DIFF|{expected_fname}|{expected_tensor.shape}|{expected_tensor.mean()}|{synced_tensor.shape}|{synced_tensor.mean()}"

def chatlearn_compare(pre_sync_dir, post_sync_dir):
    total = 0
    not_exists = 0
    records = []
    tasks = []
    for tp_rank in os.listdir(post_sync_dir):
        for param in os.listdir(os.path.join(post_sync_dir, tp_rank)):
            synced_fname = os.path.join(post_sync_dir, tp_rank, param)
            expected_fname = os.path.join(pre_sync_dir, tp_rank, param)
            message = f"{tp_rank}|{param}"
            total += 1
            if not os.path.exists(expected_fname):
                records.append(f"NOT_EXISTS|{message}")
                not_exists += 1
                continue
            tasks.append((expected_fname, synced_fname))
    with ThreadPoolExecutor(max_workers=os.cpu_count()/2) as executor:
        futures = []
        for task in tasks:
            futures.append(executor.submit(check_tensor_value, *task))

        for _future in concurrent.futures.as_completed(futures):
            try:
                record = _future.result()
                print(f"record: {record}", flush=True)
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Parameter sync thread generated an exception: {e}") # pylint: disable=raise-missing-from
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root dir to check the dumped parameters")
    args = parser.parse_args()
    chatlearn_compare(os.path.join(args.root_dir, "prev_sync"), os.path.join(args.root_dir, "post_sync"))
