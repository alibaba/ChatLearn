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

import argparse
import os
import torch

def chatlearn_compare(expected_dir, actural_dir):
    total = 0
    diff = 0
    not_exists = 0
    for tp_rank in os.listdir(actural_dir):
        for param in os.listdir(os.path.join(actural_dir, tp_rank)):
            actual_fname = os.path.join(actural_dir, tp_rank, param)
            expected_fname = os.path.join(expected_dir, tp_rank, param)
            message = f"{tp_rank}|{param}"
            total += 1
            if not os.path.exists(expected_fname):
                print(f"NOT_EXISTS|{message}|NOT_EXISTS", flush=True)
                not_exists += 1
                continue
            ta = torch.load(actual_fname, map_location="cpu")
            tb = torch.load(expected_fname, map_location="cpu")
            if not torch.allclose(ta, tb):
                print(f"DIFF|{message}|{ta.shape}|{ta.mean()}|{tb.shape}|{tb.mean()}", flush=True)
            else:
                print(f"PASS|{message}")
    print(f"ALL: {all}, DIFF: {diff}, NOT_EXISTS: {not_exists}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root dir to check the dumped parameters")
    args = parser.parse_args()
    dir1 = os.path.join(args.root_dir, "before_sync_paramter")
    dir2 = os.path.join(args.root_dir, "after_sync_paramter")
    chatlearn_compare(dir1, dir2)
