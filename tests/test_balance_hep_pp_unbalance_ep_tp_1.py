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
"""
Test:
1. trainer_tp < inference_tp
2. trainer_tp can divide inference_tp
3. trainer_tp * trainer_ep == inference_tp * inference_ep
4. HEP is enabled for both trainer and inference.

Current test case: 
(dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp) = (1, 4, 1, 4, 1, 1).
"""

import os
import ray
import time

import torch
from collections import defaultdict
from tqdm import tqdm

import chatlearn
from chatlearn import TorchModule
from chatlearn.runtime.engine import Engine
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.trainer import Trainer
from chatlearn.utils import future


trainer_params = {}
inference_params = {}

# TP rank to weights
ParamsToSync_Inference = {
    0 : {
        "mlp.experts.weight_1" : [8, 8],
        "mlp.shared_experts.weight" : [4, 8]
    },
    1 : {
        "mlp.experts.weight_2" : [8, 8],
        "mlp.shared_experts.weight" : [4, 8]
    },
    2 : {
        "mlp.experts.weight_3" : [8, 8],
        "mlp.shared_experts.weight" : [4, 8]
    },
    3 : {
        "mlp.experts.weight_4" : [8, 8],
        "mlp.shared_experts.weight" : [4, 8]
    }
}

# EP rank to weights
ParamsToSync_Trainer = {
    "ep" : {
        0 : {
            "mlp.experts.weight_1" : [8, 8],
        },
        1 : {
            "mlp.experts.weight_2" : [8, 8],
        },
        2 : {
            "mlp.experts.weight_3" : [8, 8],
        },
        3 : {
            "mlp.experts.weight_4" : [8, 8],
        }
    },
    "dp" : {
        "mlp.shared_experts.weight" : [16, 8]
    }
}

class TestTorchModule(TorchModule):

    def _get_rank(self):
        return int(os.environ["RANK"])

    def get_local_param_ranks(self):
        global_rank = self._get_rank()
        rank_0 = int(global_rank % 4)
        data_modulo_expert_parallel_ranks = [rank_0, rank_0 + 4]
        return data_modulo_expert_parallel_ranks, int(global_rank // 4)

    def check_param_exists(self, names):
        return True

    def tensor_and_expert_model_parallel_size(self):
        return 4


class CustomEngine(Engine):
    """Custom Engine"""

    def __init__(self,
                 policy: TestTorchModule,
                 policy_trainer: TestTorchModule):
        def env_compute_flow(batch):
            policy_out = policy.forward_step(batch)
            return policy_out

        def trainer_compute_flow(batch):
            policy_trainer.train_step(batch)

        env = Environment(env_compute_flow)
        trainer = Trainer(trainer_compute_flow)
        super().__init__(env, trainer)
        self.set_parameter_sync(policy_trainer, policy)


class PolicyModel(TestTorchModule):

    def model_setup(self):
        super().model_setup()
        tmp = {}
        for name, shape in ParamsToSync_Inference[self.tensor_parallel_rank()].items():
            tensor = torch.rand(shape).cuda()
            tmp[name] = tensor
        global inference_params
        inference_params[f"{self.expert_parallel_rank()}_{self.tensor_parallel_rank()}_{self.pipeline_parallel_rank()}"] = tmp
        self._named_parameters = tmp

    def get_parameter_names(self, requires_grad=True):
        return list(ParamsToSync_Inference[self.tensor_parallel_rank()].keys())

    @property
    def named_parameters(self):
        """
        :meta private:
        """
        return self._named_parameters

    def get_parameter(self, name):
        return inference_params[f"{self.expert_parallel_rank()}_{self.tensor_parallel_rank()}_{self.pipeline_parallel_rank()}"][name]

    def set_sync_parameters(self, trainable_param_names, pipe_stage=0, parameters_to_sync=None):
        if parameters_to_sync is None:
            parameters_to_sync = self._parameters_to_sync
        all_params = []
        for name in trainable_param_names:
            tensor = self.named_parameters[name]
            all_params.append((name, tensor))
        parameters_to_sync[pipe_stage] = all_params

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        query = data["query"]
        bs = query.size(0)
        data["policy_out"] = torch.ones([bs, 1024]).cuda()
        return data

    @property
    def data_parallel_size(self):
        return 2

    @property
    def data_parallel_rank(self):
        return int(self._get_rank() // 4)

    def tensor_parallel_rank(self):
        return int(self._get_rank() % 4)

    def expert_parallel_rank(self):
        return 0

    def pipeline_parallel_rank(self):
        return 0


class PPOPolicy(TestTorchModule):

    def model_setup(self):
        super().model_setup()
        tmp = {}

        for name, shape in ParamsToSync_Trainer["ep"][self.expert_parallel_rank()].items():
            tensor = torch.rand(shape).cuda()
            tmp[name] = tensor

        for name, shape in ParamsToSync_Trainer["dp"].items():
            tensor = torch.arange(0, shape[0] * shape[1]).reshape(shape).to(torch.float32).cuda() # params should be identical across different ranks in current setting.
            tmp[name] = tensor

        global trainer_params
        trainer_params[f"{self.expert_parallel_rank()}_{self.tensor_parallel_rank()}_{self.pipeline_parallel_rank()}"] = tmp
        self._named_parameters = tmp


    def get_parameter_names(self, requires_grad=True):
        return list(ParamsToSync_Trainer["ep"][self.expert_parallel_rank()].keys()) + list(ParamsToSync_Trainer["dp"].keys())

    def build_pipeline_layer_name_mapping(self, num_target_pipe_stage, target_pipe_rank, tgt_layer_offset, requires_grad=True):
        dst_src_mappings = {}

        src_ep_names = ParamsToSync_Trainer["ep"][self.expert_parallel_rank()].keys()
        for key, value in zip(src_ep_names, src_ep_names):
            dst_src_mappings[key] = value

        src_dp_names = ParamsToSync_Trainer["dp"].keys()
        for key, value in zip(src_dp_names, src_dp_names):
            dst_src_mappings[key] = value
        return dst_src_mappings

    @property
    def named_parameters(self):
        """
        :meta private:
        """
        return self._named_parameters

    def get_parameter(self, name):
        return trainer_params[f"{self.expert_parallel_rank()}_{self.tensor_parallel_rank()}_{self.pipeline_parallel_rank()}"][name]

    @property
    def data_parallel_size(self):
        return 8

    @property
    def data_parallel_rank(self):
        return self._get_rank()

    def tensor_parallel_rank(self):
        return 0

    def expert_parallel_rank(self):
        return int(self._get_rank() % 4)

    def pipeline_parallel_rank(self):
        return 0


# tuples: (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp)
tuples = (1, 4, 1,
          4, 1, 1)

chatlearn.init()
for _, model_config in chatlearn.get_args().models.items():
    model_config.num_gpu = 8
chatlearn.get_args().models['policy'].expert_model_parallel_size = tuples[0]
chatlearn.get_args().models['policy'].tensor_model_parallel_size = tuples[1]
chatlearn.get_args().models['policy'].pipeline_model_parallel_size = tuples[2]
chatlearn.get_args().models['ppo_policy'].expert_model_parallel_size = tuples[3]
chatlearn.get_args().models['ppo_policy'].tensor_model_parallel_size = tuples[4]
chatlearn.get_args().models['ppo_policy'].pipeline_model_parallel_size = tuples[5]

chatlearn.get_args().runtime_args.colocation = [["policy", "ppo_policy"]]

policy = PolicyModel("policy")
ppo_policy = PPOPolicy("ppo_policy")

os.environ['QWEN_VERSION'] = "qwen_moe_v1" # enable ParameterSyncGroupwithHEP

engine = CustomEngine(policy, ppo_policy)
engine.setup()
param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

assert param_sync_group.ep_num_mapping == tuples[0] / tuples[3]
assert param_sync_group.tp_num_mapping == tuples[1] // tuples[4]
assert param_sync_group.hep_num_mapping == 1

actor2rank = param_sync_group.actor2rank

# test for actors to create_group_experts_regrouping
send_actors = []
for actors in param_sync_group.send_actors_to_allgather_routed_experts:
    send_actors.append([actor2rank[actor] for actor in actors])
assert send_actors == [[4, 5, 6, 7], [0, 1, 2, 3]]

# Judge routed experts and parameters except routed experts
comm_pair_routed_experts = []
comm_pair_stage_1 = []
comm_pair_stage_2 = []

for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_for_routed_experts.items():
    for dst_rank in dst_ranks:
        comm_pair_routed_experts.append((actor2rank[src_rank], actor2rank[dst_rank]))

for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings.items():
    for dst_rank in dst_ranks:
        comm_pair_stage_1.append((actor2rank[src_rank], actor2rank[dst_rank]))

for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_stage2.items():
    for dst_rank in dst_ranks:
        comm_pair_stage_2.append((actor2rank[src_rank], actor2rank[dst_rank]))

# The replica iter for routed experts will be reversed because num_src_tensor_parallel == 1 
# and src_model is colocated with dst_model (Please see parameter_sync.py#L257-260).
assert comm_pair_routed_experts == [
    (4, 8), (5, 9), (6, 10), (7, 11), (0, 12), (1, 13), (2, 14), (3, 15)
]
assert comm_pair_stage_1 == [(0, 8), (1, 12)]
assert comm_pair_stage_2 == [(8, 9), (8, 10), (8, 11), (12, 13), (12, 14), (12, 15)]

print(f"pass test_case (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp): {tuples}")

engine.model_manager.sync_parameters(requires_grad=False, validate=True)
print(f"pass parameter sync validation for hyper expert parallel.")
