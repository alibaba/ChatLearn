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
"""Initialize env."""

from contextlib import closing
from datetime import timedelta
import os
import socket

import torch

from rlhf.arguments import parse_args


def get_host_ip():
    """
    get ip address in current node
    """
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    return ip_addr


def find_free_port():
    """
    find a free port
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as handle:
        handle.bind(('', 0))
        handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return handle.getsockname()[1]


def _get_active_model(rank, model_ranks):
    """
    get the model to be executed, and set env info
    """
    rank_offset = 0
    for i, (model, ranks) in enumerate(model_ranks):
        if rank in ranks:
            is_first = rank == ranks[0]
            # TODO(sayang): set it as a property
            model.is_first_rank = is_first
            model.global_ranks = ranks
            model.global_rank = rank
            world_size = len(ranks)
            model.world_size = world_size
            model.group_id = i
            model.rank_offset = rank_offset
            model.rank = rank - rank_offset
            model.active = True
            return model
        rank_offset += len(ranks)
    raise RuntimeError("rank {} not in rank list {}".format(rank, model_ranks))


def init_process_group(models, shared_path):
    """
    param models: models that contain device information
    param shared_path: shared path for broadcasting port
    """
    models = [model for model in models if model is not None]
    model_ranks = []
    rank_offset = 0

    # for model, device_count in model_to_device:
    for model in models:
        model_ranks.append((model, [i+rank_offset for i in range(model.device_count)]))
        rank_offset += model.device_count
    rank = int(os.environ["RANK"])
    model = _get_active_model(rank, model_ranks)
    if model.group_id > 0:
        master_ip = get_host_ip()
        # TODO(sayang): need to clear this file every run if it already exists
        store_path = os.path.join(shared_path, "init_group{}".format(model.group_id))
        if model.is_first_rank:
            assert not os.path.exists(store_path), "need to clear {} every run if it already exists".format(store_path)
        store = torch.distributed.FileStore(store_path, model.world_size)
        if model.is_first_rank:
            # find another available master port for group i
            master_port = find_free_port()
            # broadcast to other ranks
            store.set('port', str(master_port))
            print('rank{}: find a free port {}'.format(rank, master_port), flush=True)
        else:
            master_port = int(store.get('port'))
            print('rank{}: get a free port {}'.format(rank, master_port), flush=True)
    else:
        master_ip = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])
    model.master_ip = master_ip
    model.master_port = master_port

    os.environ["RANK"] = str(model.rank)
    os.environ["WORLD_SIZE"] = str(model.world_size)
    os.environ["MASTER_ADDR"] = model.master_ip
    os.environ["MASTER_PORT"] = str(model.master_port)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    print("start init {} rank: {}, local rank: {}, world_size: {}, addr: {}, port: {}".format(model.name, os.environ["RANK"],
        local_rank,
        os.environ["WORLD_SIZE"],
        os.environ["MASTER_ADDR"],
        os.environ["MASTER_PORT"]), flush=True)

    torch.distributed.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=10))
    print("Initialize {} done. WORLD_SIZE: {}, rank: {}".format(model.name,
          torch.distributed.get_world_size(),
          torch.distributed.get_rank()), flush=True)



def init(models):
    """
    Initialize RLHF env, including
    1. init_process_group for distributed
    2. ...
    """
    args = parse_args()
    set_global_args(args)
    init_process_group(models, args.shared_path)
