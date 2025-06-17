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
"""Sync parameters"""
import numpy as np
import concurrent.futures
import traceback
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import List
from queue import PriorityQueue

import torch
from tqdm import tqdm

from chatlearn.launcher.initialize import patch_ray
from chatlearn.utils import future
from chatlearn.utils import utils
from chatlearn.utils.constant import PARAM_SYNC_COMM_TYPE
from chatlearn.utils.constant import ROUTED_EXPERT_REGROUPING_COMM_TYPE
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.utils.utils import execute_in_parallel
from chatlearn.utils.timer import Timers
from . import get_synchronizer

patch_ray()

from typing import *
from enum import Enum
class TensorMeta:
    dtype: torch.dtype = None
    offset: Tuple[int] = None
    global_shape: Tuple[int] = None
    local_shape: Tuple[int] = None
    is_sparse: bool = False

class MappingType(Enum):
    ...

class MegatronVLLMMapper:
    """
        The Mapper for Megatron to vLLM sync. In each remote Megatron Actor, 
        the method of this class is called to generate the parameter mapping 
        between src and dst.
    """
    def __init__(self, model: MegatronModule):
        ...

    def generate_sync_mapping(self):
        """
            Generate the synchronization mapping of this local rank. The return dict
            is the plan including all local parameters to be synchronized in the 
            following format:
            {
                MappingType.TypeA: {
                    (param_idx, TensorMeta): (param_idy, TensorMeta),
                    ...
                },
                ...
            }
            where `param_idx` represents the parameter id of the parameter in the src model,
            `param_idy` represents id from the dst model. Each idx/idy can show multiple times,
            but (param_id*, TensorMeta) is unique in the global view.
        """
        pass

    def dump_sync_mapping(self, folder_path: str, sync_mapping: Dict):
        """dump the generayed sync mapping to the given folder path in JSON format.

        Args:
            folder_path (str): The folder path to dump the sync mapping.
            sync_mapping (Dict): The sync mapping to be saved.
        """
        pass


class GenericSynchronizer:
    """
        Execute parameter synchronization based on the given sync plan. For simplicity,
        the current implementation claims that each actor should either be a parameter 
        provider (through IPC Handle) P or a rank C in the global communication group.

        In the synchronization process, (1) all Ps will firstly transform and bucketize 
        their parameters, and (2) send this bucket through IPC to their colocated Cs. 
        Then (3) all Cs will start an all-to-all Op to send-recv buckets from each other, 
        and finally (4) the received buckets will be extracted and remerged to update 
        local parameters.

        In each iteration, this synchronizer applys the following operation
        for each parameter in the bucket:
        >>>    index(dst_tensor, tensormeta_dst) = transform_mapping[mapping_type](
        >>>        index(src_tensor, tensormeta_src)
        >>>    )

        Therefore, the synchronizer does not require the full tensor to be synchronized
        in one iteration.
    """

    def set_plan(self, sync_plan):
        ...

    pass

class MegatronVLLMSyncPlanner:
    """Generate the sync plan based on the given sync mapping and ModelParallel setting.
    The plan is a mapping of send(recv) sharded parameters in one iteration:

    # send plan of rank i, recv plan also has the following format but 
    # assure len(key) = 1, i.e., only from one source
    [
        # iter 0
        [
            # List[remote_rank]: BucketInfo, each bucket is either a bucket of dense 
            # parameters or a bucket of sparse parameters
            (remote_rank1, remote_rank2, ...): [(local_param_id, TensorMeta), ...],
            (...): [(local_param_id, TensorMeta), ...]
        
        ]
        # iter 1
        ...
    ]
    """
    def __init__(self, sync_mapping, model_parallel_size):
        ...

    def make_plan(self):
        """
            Make a sync plan on N gpus where:
            Megatron:
                Dense: TP x PP = N // (CP x DP)
                MoE: ETP x EP x PP = N // EDP
            vLLM:
                Dense: PP x TP = N // DP
                MoE: EP x PP = N
        """
        # NOTE: for dense param, balance payloads across CP x DP group, i.e.,
        # if we have M tensors of same size, ranks of CPxDPy will send 
        # M // (CP x DP) tensors independently.

        # NOTE: for sparse param, balance payloads across EDP group
        ...

    @staticmethod
    def approximate_bin_packing(items: np.array, K: int) -> List[List[int]]:
        """Packing N items into K bins and make the payloads 
            of each bin as close as possible.

        Args:
            items (np.array): The sizes of each item
            K (int): The num of buckets.

        Returns:
            List[List[int]]: The item index of each bucket.
        """
        bins = np.zeros(K)
        results = [list() for _ in range(K)]
        for idx in items.argsort()[::-1]:
            bins[bins.argmin()] += items[idx]
            results[bins.argmin()].append(idx)
        return results


class ParameterSyncGroup:

    def __init__(self, src_model: DistModel, dst_model: DistModel, group_name: str):
        """Manage Parameter Synchronization between source and destination models.

        Args:
            src_model (DistModel): The source distmodel, only MegatronModel is supported.
            dst_model (DistModel): The destination distmodel, only vLLM backend is supported.
            group_name (str): The tag of this parameter sync group.
        """




        ...


    def collect_parameter_metadata(self, model: DistModel) -> Dict[int, Dict[int, TensorMeta]]:
        """Collect parameter metadata from model.

        Args:
            model (DistModel): The model to collect parameter metadata.
        
        Returns:
            Dict[int, Dict[int, TensorMeta]]: The parameter metadata with the following 
            format:
            {
                rank: {
                    param_id: TensorMeta
                }
            }
        """

        pass

    def generate_global_param_ids(self, model: DistModel):
        """According to the layout of ModelParallel in Megatron and vLLM, any parameter 
        Tensor can be identified by (ep_rank, pp_rank, local_weight_name). This function
        will generate a global parameter id for each Tensor in the state dict of the model, 
        even if the Tensor will not be synced.

        Args:
            model (DistModel): The model to generate global parameter ids.
        """
        # Megatron: Dense (TP-CP-DP-PP)  MoE (ETP-EP-EDP-PP)
        # vLLM: Dense (DP-PP-TP), EP group is DP-TP group, i.e., ETP=EDP=1
        raise NotImplementedError()
    
    def validate_sync_mapping(self, sync_mappings: List[Dict]):
        """
            Check whether the merged sync mapping from all source actors meets the metadata
            collected by `collect_parameter_metadata`.
        
        Args:
            sync_mappings (List[Dict]): The sync mappings from all source actors.

        """
        pass


    
    def sync(self, *args, **kwargs):


        pass