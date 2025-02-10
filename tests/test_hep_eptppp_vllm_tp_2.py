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
2. trainer_pp > inference_pp
2. trainer_ep > 1 while inference_ep = 1
3. HEP is enabled for trainer but disabled for inference.

Current test case: 
(dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp) = (1, 4, 1, 2, 2, 2).
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
from chatlearn.schedule.resource_manager import ResourceManager
from chatlearn.schedule.model_manager import ModelManager
from chatlearn.synchronizer.parameter_sync import ParameterSyncGroupwithHEP

trainer_params = {}
inference_params = {}

# TP rank to weights
inf_name_shape_dict = {
    "layers.0.mlp.experts.dense_h_to_4h.weight_0" : [1, 8, 2],
    "layers.0.mlp.experts.dense_4h_to_h.weight_0" : [1, 2, 8],
    "layers.0.mlp.experts.dense_h_to_4h.weight_1" : [1, 8, 2],
    "layers.0.mlp.experts.dense_4h_to_h.weight_1" : [1, 2, 8],
    "layers.0.mlp.experts.dense_h_to_4h.weight_2" : [1, 8, 2],
    "layers.0.mlp.experts.dense_4h_to_h.weight_2" : [1, 2, 8],
    "layers.0.mlp.experts.dense_h_to_4h.weight_3" : [1, 8, 2],
    "layers.0.mlp.experts.dense_4h_to_h.weight_3" : [1, 2, 8],
    "layers.0.mlp.shared_experts.dense_h_to_4h.weight" : [4, 16],
    "layers.0.mlp.shared_experts.dense_4h_to_h.weight" : [16, 4],
    "layers.1.mlp.experts.dense_h_to_4h.weight_0" : [1, 8, 2],
    "layers.1.mlp.experts.dense_4h_to_h.weight_0" : [1, 2, 8],
    "layers.1.mlp.experts.dense_h_to_4h.weight_1" : [1, 8, 2],
    "layers.1.mlp.experts.dense_4h_to_h.weight_1" : [1, 2, 8],
    "layers.1.mlp.experts.dense_h_to_4h.weight_2" : [1, 8, 2],
    "layers.1.mlp.experts.dense_4h_to_h.weight_2" : [1, 2, 8],
    "layers.1.mlp.experts.dense_h_to_4h.weight_3" : [1, 8, 2],
    "layers.1.mlp.experts.dense_4h_to_h.weight_3" : [1, 2, 8],
    "layers.1.mlp.shared_experts.dense_h_to_4h.weight" : [4, 16],
    "layers.1.mlp.shared_experts.dense_4h_to_h.weight" : [16, 4]
}
ParamsToSync_Inference = {
    0 : inf_name_shape_dict,
    1 : inf_name_shape_dict,
    2 : inf_name_shape_dict,
    3 : inf_name_shape_dict
}

# PP rank to (EP rank to weights and TP rank to weights)
ParamsToSync_Trainer = {
    "pp" : {
        0 : {
            "ep" : {
                0 : {
                    "layers.0.mlp.experts.dense_h_to_4h.weight_0" : [1, 8, 8],
                    "layers.0.mlp.experts.dense_4h_to_h.weight_0" : [1, 8, 8]
                },
                1 : {
                    "layers.0.mlp.experts.dense_h_to_4h.weight_1" : [1, 8, 8],
                    "layers.0.mlp.experts.dense_4h_to_h.weight_1" : [1, 8, 8]
                },
                2 : {
                    "layers.0.mlp.experts.dense_h_to_4h.weight_2" : [1, 8, 8],
                    "layers.0.mlp.experts.dense_4h_to_h.weight_2" : [1, 8, 8]
                },
                3 : {
                    "layers.0.mlp.experts.dense_h_to_4h.weight_3" : [1, 8, 8],
                    "layers.0.mlp.experts.dense_4h_to_h.weight_3" : [1, 8, 8]
                }
            },
            "tp" : {
                0 : {
                    "layers.0.mlp.shared_experts.dense_h_to_4h.weight" : [8, 16],
                    "layers.0.mlp.shared_experts.dense_4h_to_h.weight" : [16, 8]
                },
                1 : {
                    "layers.0.mlp.shared_experts.dense_h_to_4h.weight" : [8, 16],
                    "layers.0.mlp.shared_experts.dense_4h_to_h.weight" : [16, 8]
                }
            }
        },
        1 : {
            "ep" : {
                0 : {
                    "layers.1.mlp.experts.dense_h_to_4h.weight_0" : [1, 8, 8],
                    "layers.1.mlp.experts.dense_4h_to_h.weight_0" : [1, 8, 8]
                },
                1 : {
                    "layers.1.mlp.experts.dense_h_to_4h.weight_1" : [1, 8, 8],
                    "layers.1.mlp.experts.dense_4h_to_h.weight_1" : [1, 8, 8]
                },
                2 : {
                    "layers.1.mlp.experts.dense_h_to_4h.weight_2" : [1, 8, 8],
                    "layers.1.mlp.experts.dense_4h_to_h.weight_2" : [1, 8, 8]
                },
                3 : {
                    "layers.1.mlp.experts.dense_h_to_4h.weight_3" : [1, 8, 8],
                    "layers.1.mlp.experts.dense_4h_to_h.weight_3" : [1, 8, 8]
                }
            },
            "tp" : {
                0 : {
                    "layers.1.mlp.shared_experts.dense_h_to_4h.weight" : [8, 16],
                    "layers.1.mlp.shared_experts.dense_4h_to_h.weight" : [16, 8]
                },
                1 : {
                    "layers.1.mlp.shared_experts.dense_h_to_4h.weight" : [8, 16],
                    "layers.1.mlp.shared_experts.dense_4h_to_h.weight" : [16, 8]
                }
            }
        }
    }
    
}



class TestTorchModule(TorchModule):

    def _get_rank(self):
        return int(os.environ["RANK"])

    def get_local_param_ranks(self):
        global_rank = self._get_rank()
        data_modulo_expert_parallel_ranks = [global_rank]
        return data_modulo_expert_parallel_ranks, 0

    def check_param_exists(self, names):
        return True

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

    def _create_remote_models(self):
        """
        :meta private:
        """
        resource_manager = ResourceManager(self._models)
        self.model_manager = MockModelManagerHEP(self._models, resource_manager, self.global_args)
        for src_model, dst_model in self._param_sync_pairs:
            self.model_manager.set_parameter_sync(src_model, dst_model)
        self.model_manager.remote()
        self.remote_models = self.model_manager.dist_models
        self.named_models = {model.name: model for model in self.remote_models}


class MockModelManagerHEP(ModelManager):

    def build_parameter_group(self):
        for src_model, dst_model in self._parameter_sync_model_pair:
            group_name = self._get_group_name(src_model, dst_model)
            sync_frequency = self._get_sync_frequency(dst_model)
            sync_group = MockParameterSyncGroupwithHEP(
                self._name2distmodel[src_model.name],
                self._name2distmodel[dst_model.name],
                group_name,
                sync_frequency,
                self.error_signal
            )
            self.parameter_sync_groups[group_name] = sync_group


class MockParameterSyncGroupwithHEP(ParameterSyncGroupwithHEP):

    def setup_rank_mapping(self):
        self.tp_num_mapping = self.num_dst_tensor_parallel // self.num_src_tensor_parallel
        self.ep_num_mapping = self.num_dst_expert_parallel / self.num_src_expert_parallel
        self.hep_num_mapping = self.num_dst_hyper_expert_parallel / self.num_src_hyper_expert_parallel

        self.build_rank_mapping_for_ep(add_recv_actor_fn=self.add_recv_actor_for_routed_experts)
        self.build_rank_mapping_for_params_except_routed_expert()


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

    def tensor_and_expert_model_parallel_size(self):
        return 4


class PPOPolicy(TestTorchModule):

    def model_setup(self):
        super().model_setup()
        tmp = {}
        for name, shape in ParamsToSync_Trainer["pp"][self.pipeline_parallel_rank()]["ep"][self.expert_parallel_rank()].items():
            tensor = torch.rand(shape).cuda()
            tmp[name] = tensor

        for name, shape in ParamsToSync_Trainer["pp"][self.pipeline_parallel_rank()]["tp"][self.tensor_parallel_rank()].items():
            # params should be identical across different ranks in current setting.
            offset = self.tensor_parallel_rank()
            tensor = torch.arange(0 + offset, shape[0] * shape[1] + offset).reshape(shape).to(torch.float32).cuda()
            tmp[name] = tensor

        global trainer_params
        trainer_params[f"{self.expert_parallel_rank()}_{self.tensor_parallel_rank()}_{self.pipeline_parallel_rank()}"] = tmp
        self._named_parameters = tmp

    def get_parameter_names(self, requires_grad=True):
        return list(ParamsToSync_Trainer["pp"][self.pipeline_parallel_rank()]["ep"][self.expert_parallel_rank()].keys()) \
            + list(ParamsToSync_Trainer["pp"][self.pipeline_parallel_rank()]["tp"][self.tensor_parallel_rank()].keys())

    def build_pipeline_layer_name_mapping(self, num_target_pipe_stage, target_pipe_rank, tgt_layer_offset, requires_grad=True):
        dst_src_mappings = {}

        src_ep_names = ParamsToSync_Trainer["pp"][self.pipeline_parallel_rank()]["ep"][self.expert_parallel_rank()].keys()
        for key, value in zip(src_ep_names, src_ep_names):
            dst_src_mappings[key] = value

        src_tp_names = ParamsToSync_Trainer["pp"][self.pipeline_parallel_rank()]["tp"][self.tensor_parallel_rank()].keys()
        for key, value in zip(src_tp_names, src_tp_names):
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
        return 1

    @property
    def data_parallel_rank(self):
        return 0

    def tensor_parallel_rank(self):
        return int(self._get_rank() % 2)

    def expert_parallel_rank(self):
        return int(self._get_rank() % 4 // 2)

    def pipeline_parallel_rank(self):
        if self._get_rank() < 4:
            return 0
        return 1

    def tensor_and_expert_model_parallel_size(self):
        return 4

def test_hep_eptppp_vllm_tp_dst_ep1_tp4_pp1_src_ep2_tp2_pp2():
    # tuples: (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp)
    tuples = (1, 4, 1,
              2, 2, 2)

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

    # Judge num mappings
    assert param_sync_group.ep_num_mapping == tuples[0] / tuples[3]
    assert param_sync_group.tp_num_mapping == tuples[1] // tuples[4]

    # Judge alltoall actors
    alltoall_actors = param_sync_group.send_actors_to_regroup_routed_experts
    actor2rank = param_sync_group.actor2rank

    assert param_sync_group._comm_type_to_regroup_routed_experts == "alltoall"
    assert len(alltoall_actors) == 2
    assert len(alltoall_actors[0]) == 4 # prev 4 src ranks should all-to-all routed experts
    assert len(alltoall_actors[1]) == 4 # last 4 src ranks should all-to-all routed experts
    assert len(actor2rank) == 16 # all of the 16 actors should have rank
    assert len(set(list(actor2rank.values()))) == len(actor2rank) # all ranks should be unique

    alltoall_actor_ranks = []
    for actor_list in alltoall_actors:
        alltoall_actor_ranks.append([])
        for actor in actor_list:
            alltoall_actor_ranks[-1].append(actor2rank[actor])

    assert alltoall_actor_ranks == [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Judge src->dst rank mappings
    comm_pairs_for_routed_experts = []
    comm_pairs_for_except_routed_experts_stage1 = []
    comm_pairs_for_except_routed_experts_stage2 = []

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_for_routed_experts.items():
        for dst_rank in dst_ranks:
            comm_pairs_for_routed_experts.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings.items():
        for dst_rank in dst_ranks:
            comm_pairs_for_except_routed_experts_stage1.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst_rank in dst_ranks:
            comm_pairs_for_except_routed_experts_stage2.append((actor2rank[src_rank], actor2rank[dst_rank]))

    assert comm_pairs_for_routed_experts == [
        (0, 8), (0, 12), (1, 9), (1, 13),
        (2, 10), (2, 14), (3, 11), (3, 15),
        (4, 8), (4, 12), (5, 9), (5, 13),
        (6, 10), (6, 14), (7, 11), (7, 15) 
    ], f"{comm_pairs_for_routed_experts}"
    assert comm_pairs_for_except_routed_experts_stage1 == [
        (0, 9), (1, 11), (4, 8), (5, 10),
        (2, 13), (3, 15), (6, 12), (7, 14)
    ], f"{comm_pairs_for_except_routed_experts_stage1}"
    assert comm_pairs_for_except_routed_experts_stage2 == [
        (8, 9), (9, 8), (10, 11), (11, 10),
        (12, 13), (13, 12), (14, 15), (15, 14)
    ], f"{comm_pairs_for_except_routed_experts_stage2}"

    print(f"pass test_case (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp): {tuples}")

    engine.model_manager.sync_parameters(requires_grad=False, validate=False)
    print(f"pass parameter sync validation for hyper expert parallel and vllm when gpus collide.")

test_hep_eptppp_vllm_tp_dst_ep1_tp4_pp1_src_ep2_tp2_pp2()
