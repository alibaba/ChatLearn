# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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
"""fsdp to vllm parameter sync group"""
from itertools import chain

import ray
from chatlearn.utils import future

from .base_parameter_sync import BaseParameterSyncGroup


class FSDPParameterSyncGroup(BaseParameterSyncGroup):
    """fsdp to vllm or sglang parameter sync group"""

    def sync(self, dryrun: bool=False):
        """Perform parameter synchronization on this group. If `dryrun` is True,
        some initialization will be excuted and no actual synchronization 
        will be done.

        Args:
            dryrun (bool, optional): Whether to run in dryrun mode. 
            Defaults to False.
        """
        if dryrun:
            return
        src_model, dst_model = self.src_model, self.dst_model
        # NOTE: for each source actor, find its colocated dst actor.
        src_gpus = future.wait(src_model.call_func_on_all_workers('get_gpu_info'), return_output=True)
        dst_gpus = future.wait(dst_model.call_func_on_all_workers('get_gpu_info'), return_output=True)
        src_rank_to_gpu_id = dict(zip(chain.from_iterable(src_model.all_actor_ids), src_gpus))
        gpu_id_to_dst_rank = dict(zip(dst_gpus, chain.from_iterable(dst_model.all_actor_ids)))
        src_rank_to_dst_rank = {
            src_rank: gpu_id_to_dst_rank[src_gpu_id]
            for src_rank, src_gpu_id in src_rank_to_gpu_id.items()
        }

        param_name_list = ray.get(src_model.get_actor(0).get_fsdp_param_name.remote())
        for param_name in param_name_list:
            refs = []
            for src_rank, dst_rank in src_rank_to_dst_rank.items():
                src_actor = src_model.get_actor(src_rank)
                dst_actor = dst_model.get_actor(dst_rank)
                reduce_data_ref = src_actor.get_weight_ipc_handles_by_name.remote(param_name)
                ref = dst_actor.update_weights_from_ipc_handles.remote(reduce_data_ref)
                refs.append(ref)
            future.wait(refs, return_output=True)
