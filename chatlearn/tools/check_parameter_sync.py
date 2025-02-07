import os
import torch
import shutil
import argparse
from collections import defaultdict


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
