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
"""get remote object"""

import ray

from chatlearn.utils.logger import logging_tqdm
from chatlearn.utils.utils import flatten


def check_nested_2_level_list(refs):
    """
    Checks if a list is a nested list with a nested level of 2.
    e.g.
    [[ref0, ref1], [ref2, ref3]] returns True, [2, 2]
    [ref0, ref1] returns False, -1
    [[ref0], [ref1, ref2]] returns True, [1, 2]

    Returns a tuple containing two elements:
    - A boolean value indicating if the list is a nested 2-level list
    - A list of integers containing the length of each sublist
    """
    sublist_lens = []
    for sublist in refs:
        if isinstance(sublist, list):
            if len(sublist) == 0:
                sublist_lens.append(0)
            else:
                if isinstance(sublist[0], ray.ObjectRef):
                    sublist_lens.append(len(sublist))
                else:
                    return False, None
        else:
            return False, None
    return True, sublist_lens


def wait(refs, desc=None, return_output=False):
    """
    wait until all computation finish
    """
    if isinstance(refs, ray.ObjectRef):
        ret = ray.get(refs)
        return ret if return_output else None
    if len(refs) == 0:
        return
    nested2, sublist_lens = check_nested_2_level_list(refs)
    refs = flatten(refs)
    if desc is not None:
        total = len(refs) if not nested2 else len(sublist_lens)
        pbar = logging_tqdm(total=total, desc=desc)
    i = 0
    wait_refs = refs.copy()
    while wait_refs:
        num_returns = 1 if not nested2 else sublist_lens[i]
        done, wait_refs = ray.wait(wait_refs, num_returns=num_returns)
        i += 1
        if desc is not None:
            done_size = len(done) if not nested2 else 1
            pbar.update(done_size)
    if return_output:
        outputs = ray.get(refs)
    if desc is not None:
        pbar.close()
    if return_output:
        return outputs


def get(data):
    """get remote data"""
    if isinstance(data, (list, tuple)):
        dtype = type(data)
        ret = dtype(get(item) for item in data)
        return ret
    if isinstance(data, dict):
        return {key: get(value) for key, value in data.items()}
    while isinstance(data, ray.ObjectRef):
        data = ray.get(data)
    return data
