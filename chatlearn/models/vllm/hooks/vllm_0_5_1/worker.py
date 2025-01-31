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
"""Hooks of vllm-0.5.1 worker_base to remove metadata broadcasting."""


import inspect

# pylint: disable=unused-import,wildcard-import
from vllm.worker import worker_base


source = inspect.getsource(worker_base.LocalOrDistributedWorkerBase.execute_model)
if 'self.do_metadata_broadcast' in source:
    from vllm.worker.worker_base import WorkerInput
    from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SamplerOutput)
    from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase
    from vllm.distributed import broadcast_tensor_dict, get_pp_group
    from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        assert self.is_driver_worker
        if self.is_driver_worker:
            if execute_model_req is None:
                return None

            worker_input: WorkerInput = self.prepare_worker_input(
                execute_model_req=execute_model_req)
            model_input: ModelRunnerInputBase = (
                self.model_runner.prepare_model_input(
                    execute_model_req.seq_group_metadata_list,
                    execute_model_req.virtual_engine,
                    execute_model_req.finished_requests_ids))
            num_steps = execute_model_req.num_steps

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict())

        output = self.model_runner.execute_model(
            model_input, self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None, intermediate_tensors,
            num_steps)

        if not get_pp_group().is_last_rank:
            get_pp_group().send_tensor_dict(output.tensors)
            return [None]

        # Worker only supports single-step execution. Wrap the output in a
        # list to conform to interface.
        return output

    worker_base.LocalOrDistributedWorkerBase.execute_model =  execute_model
