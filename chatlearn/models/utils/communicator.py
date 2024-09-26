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

import cupy
import torch

from ray.util.collective.collective_group import nccl_util

class Communicator:

    def ipc_send(self, tensor):
        if tensor.device.type == "cuda":
            torch.cuda.synchronize()
        # workaround for bfloat16, as cupy not support bfloat now
        # https://github.com/cupy/cupy/issues/7527
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.view(dtype=torch.uint8)
        ndarry = cupy.asarray(tensor)
        print(f"send data ptr {tensor.data_ptr()}")
        handle = cupy.cuda.runtime.ipcGetMemHandle(ndarry.data.ptr)
        return handle

    def ipc_recv(self, device_ptr, tensor): #num_bytes, shape, dtype, device):
        ndarry = cupy.asarray(tensor)
        if tensor.dtype == torch.bfloat16:
            tensor_view = tensor.view(dtype=torch.uint8)
        else:
            tensor_view = tensor
        num_bytes = tensor_view.nelement()*tensor_view.element_size()

        handle = cupy.cuda.runtime.ipcOpenMemHandle(device_ptr)
        m = cupy.cuda.UnownedMemory(handle, ndarry.size, self)
        m_ptr = cupy.cuda.MemoryPointer(m, 0)
        print('recv_memory ptr', m_ptr, flush=True)
        if tensor.dtype == torch.bfloat16:
            arr = cupy.ndarray(tensor_view.shape, dtype=cupy.uint8, memptr=m_ptr)
            out = torch.as_tensor(arr, device=tensor.device, dtype=nccl_util.TORCH_NUMPY_DTYPE_MAP[torch.uint8]).view(tensor.dtype)
        else:
            out = torch.as_tensor(cupy.ndarray(tensor_view.shape, dtype=nccl_util.TORCH_NUMPY_DTYPE_MAP[tensor.dtype], memptr=m_ptr))
        # tensor = torch.tensor(cupy.ndarray(tensor.shape, dtype, m_ptr), device=device)
        print(out)
        tensor.data.copy_(out)
        return tensor

    def ipc_send_parameter(self, group_name, pipe_stage):
        # tensor = self._named_parameters_to_sync[pipe_stage][name]
        names = sorted(self._named_parameters_to_sync[pipe_stage].keys())
        for name in names:
            tensor = self._named_parameters_to_sync[pipe_stage][name]
        # tensor = self.get_parameter(name)
            handle = self.ipc_send(tensor)
            # breakpoint()
            self.put(group_name + ":" + name, handle)

    def ipc_recv_parameter(self, group_name, pipe_stage):
        names = sorted(self._named_parameters_to_sync[pipe_stage].keys())
        for name in names:
            tensor = self._named_parameters_to_sync[pipe_stage][name]
            # tensor = self.get_parameter(name)
            handle_ref = None
            # breakpoint()
            # TODO: refine this
            if not name.startswith("module.module."):
                name = "module." + name
            while handle_ref is None:
                handle_ref = self.get(group_name + ":" + name)
            self.ipc_recv(handle_ref, tensor)
            # tensor.copy_(recv_tensor)