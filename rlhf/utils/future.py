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
"""get remote object"""

import ray

from rlhf.utils.logger import logging_tqdm
from rlhf.utils.utils import flatten


def wait(refs, desc=None, return_output=False):
    """
    wait until all computation finish
    """
    if isinstance(refs, ray.ObjectRef):
        ray.get(refs)
        return
    if len(refs) == 0:
        return
    refs = flatten(refs)
    outputs = []
    if desc is not None:
        pbar = logging_tqdm(total=len(refs), desc=desc)
    while refs:
        done, refs = ray.wait(refs)
        if desc is not None:
            pbar.update(len(done))
        res = ray.get(done[0])
        if return_output:
            outputs.append(res)
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
