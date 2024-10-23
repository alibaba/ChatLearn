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
"""base module"""

from collections import defaultdict
from functools import reduce
from itertools import cycle
import math
import operator
import os
import torch

import ray
import ray.util.collective as col
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.collective_group.nccl_collective_group import NCCLGroup
from torch.utils.data import DataLoader
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from chatlearn.data.sampler import SingleDataSampler, EpisodeDataSampler
from chatlearn.checkpoint.checkpoint_manager import CheckpointManager
from chatlearn.utils import future
from chatlearn.utils.dist_utils import bucket_tensors, coalesced_comm_dense
from chatlearn.utils.dist_utils import bucket_tensors_two_stage, coalesced_comm_dense_two_stage
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.global_vars import set_global_variables
from chatlearn.utils.logger import log_rank_0, debug_rank_0, setup_logger
from chatlearn.utils.timer import Timers
from chatlearn.utils.utils import get_host_addr
from chatlearn.launcher import dlc_utils


class BaseModule:
    """BaseModule is the base class for Base models.

    Args
    ----
    name : str
        model name
    """

    def __init__(self, name, args=None, replica_id=0):
        self.name = name
        if args is None:
            global_args = get_args()
        else:
            global_args = args
            set_global_variables(args)
        self.global_args = global_args
        args = global_args.models[name]
        self.total_gpu = args.num_gpu
        self.total_cpu = args.num_cpu
        self.gpu_per_process = args.gpu_per_process
        self.trainable = args.trainable
        self._runtime_args = self.global_args.runtime_args
        self._module_args = args
        self.replica_id = replica_id
        self.config_dir = args.config_dir
        self._is_colocate = False

        if self.total_gpu > 0:
            self._num_gpu_per_replica = (
                args.tensor_model_parallel_size
                * args.pipeline_model_parallel_size
                * args.expert_model_parallel_size
                * args.zero_size
            )
            assert self._num_gpu_per_replica <= self.total_gpu
            assert self.total_gpu % self._num_gpu_per_replica == 0
            if not self.trainable:
                self._num_replica = args.num_gpu // self._num_gpu_per_replica
            else:
                # For trainable models, perform the DP inside DistActor
                self._num_replica = 1
                self._num_gpu_per_replica = self.total_gpu
        else:
            self._num_gpu_per_replica = 0
            self._num_replica = args.num_replica

        assert self._num_replica >= 1
        self._param_ranks = None
        self._named_parameters = None
        self._param_to_name = None
        self._parameters = None
        self._coalesced_parameters = None
        self.error_signal = None
        self._rank = None
        self._world_size = None
        self._group_names = []
        self._dataloader = None
        self._eval_dataloader = None
        self._kl_coef = None
        self._padding_config = {}
        self._storage = None
        self._timers = None
        self._data_iter = None
        self._eval_data_iter = None
        self.call_funcs = []
        self.trainable_funcs = []
        self._data_ckpt_manager = None
        self._peak_memory = 0
        self._parameters_to_sync = defaultdict(list)
        self._parameters_to_send = defaultdict(list)
        self._parameters_to_recv = defaultdict(list)
        self._parameters_shape = []
        self._concat_params_dict = None
        self._to_fix_act_ordering_dict = None
        self._to_fix_qkv_ordering_dict = None
        self._to_fix_qkv_ordering_func = None
        # current compute iteration
        self._iteration = 0
        self._train_iteration = 0
        self.enable_lora = self._module_args.lora.enable_lora
        self._finalized = False
        self._resume_training = False
        self._address = dlc_utils.get_addr() if dlc_utils.in_dlc_env() else get_host_addr()
        self._is_master_node = os.environ.get("RANK", '0') == '0'
        self._logger = setup_logger(model_name=self.name, ip_addr=self._address)
        self._dummy_output = None
        self._dummy_inputs = []
        # parameter sync from src_model
        self._src_parameter_model = None
        self.profiler = None
        self._buffer_num = {}
        self._tp_division = {}
        self._num_mapping = 1
        self._sync_buffer = defaultdict(list)

    def get_sync_buffer(self):
        return self._sync_buffer

    def set_num_mapping(self, _num_mapping):
        self._num_mapping = _num_mapping

    @property
    def num_mapping(self):
        return self._num_mapping

    def set_buffer_num(self, buffer_num):
        self._buffer_num.update(buffer_num)

    def get_buffer_num(self, param_names):
        return [self._buffer_num[name] for name in param_names]

    def set_tp_division(self, tp_division):
        self._tp_division.update(tp_division)

    def get_tp_division(self, param_names):
        return [self._tp_division[name] for name in param_names]

    @property
    def is_colocate(self):
        return self._is_colocate

    def set_colocate(self, flag):
        self._is_colocate = flag

    def finalize(self):
        """
        finalize the class, any change from user after finalize will not work.

        :meta private:
        """
        self._finalized = True

    def _assert_not_finalized(self):
        """
        :meta private:
        """
        assert not self._finalized, f"{self} is finalized, any change to the class should happen before finalize."

    @property
    def runtime_args(self):
        """
        Return the arguments related to alignment training,
        the settings that are specified under the "runtime" section of the YAML configuration file.
        """
        return self._runtime_args

    @property
    def model_args(self):
        """
        Return model arguments, such as those related to Megatron,
        should be specified in a separate configuration yaml file for the model being used.
        """
        return self._module_args.args_dict

    @property
    def module_args(self):
        """
        Return module arguments. module_args include `num_gpu`, `gpu_per_process`, `model_config_file`, etc.
        """
        return self._module_args

    @property
    def parameter_sync_frequency(self):
        return self.module_args.sync_frequency

    def set_env(self, args):
        """
        set system env, private

        :meta private:
        """

    def set_error_signal(self, error_signal):
        """
        signal for handling errors

        :meta private:
        """
        self.error_signal = error_signal

    def error(self, error_msg=None):
        """
        :meta private:
        """
        future.wait(self.error_signal.set.remote(error_msg))

    def init(self):
        """
        Init env.
        """

    def setup(self):
        """
        Create model / optimizer / opt_param_scheduler / etc.
        """

    @property
    def data_ckpt_manager(self):
        """
        :meta private:
        """
        if self.runtime_args.data_checkpoint_path is not None:
            assert self._data_ckpt_manager is not None
        return self._data_ckpt_manager

    def model_setup(self):
        """
        :meta private:
        """
        self.global_args.active_module_args = self._module_args
        if self.runtime_args.data_checkpoint_path is not None:
            self._data_ckpt_manager = CheckpointManager(self, self.runtime_args.data_checkpoint_path,
                                                       self.runtime_args.max_data_ckpt_nums,
                                                       self.runtime_args.load_data_checkpoint_iteration)
            if self.runtime_args.enable_resume_training:
                meta = self._data_ckpt_manager.resume()
                if meta:
                    self._resume_training = self.runtime_args.consumed_samples > 0
                    start_episode = meta["episode"] + 1
                    self._iteration = start_episode * math.ceil(self.runtime_args.sample_per_episode / \
                        self._num_replica / self.module_args.generation_batch_size)
                    log_rank_0(f"{self.name} resume training {self._resume_training}: set start iteration to {self._iteration}", self._logger)
        self.setup()

    def forward_step(self, data, iteration):
        """
        Perform forward step for one batch.

        Args
        ----
        data : dict
            data for forward_step
        iteration : int
            local forward iteration
        
        Returns
        -------
        Dict
            A dict of results, where key is the string type, and the value is the tensor or a list,
            where the first dim of tensor or the len of list equals to batch size
        """

    def train_step(self, data, iteration):
        """
        Perform train_step for one batch, including a list of micro-batches.

        Args
        ----
        data : [Dict]
            A list of micro-batch for train_step, type of each micro-batch is dict
        iteration : int
            local train iteration
        """

    def eval_step(self, data):
        """
        Perform eval_step for one batch

        Args
        ----
            data: Dict
                Data for eval_step.

        Returns
        -------
            Dict
                A dict of results, where key is the string type, and the value is the tensor or a list,
                where the first dim of tensor or the len of list equals to batch size
        """

    def save_checkpoint(self, iteration):
        """
        Save checkpoint given iteration.

        Args
        ----
            iteration: int
                Current training iteration
        """

    def save_data_checkpoint(self, replica_id, iteration, episode_id):
        """
        Save checkpoint for dataloader.

        :meta private:
        """
        if self.data_ckpt_manager is not None:
            consumed_samples = self.runtime_args.consumed_samples
            self.data_ckpt_manager.save_checkpoint(replica_id, iteration, episode_id, consumed_samples)

    def put(self, key, data):
        """
        Put the data to shared storage.

        Args
        ----
            key: Str
                Use key to put.
            data
                data to save
        """
        self._storage.put.remote(key, data)

    def get(self, key):
        """
        Get data from shared storage using key

        Args
        ----
            key: Str
                use key to get
        """
        ref = self._storage.get.remote(key)
        return future.get(ref)

    def validate(self):
        """
        :meta private:
        """

    def before_episode(self):
        """
        Operations before one episode.
        """

    def after_episode(self):
        """
        Operations after one episode.
        """

    def build_dataset(self, train_prompts, is_eval=False):
        """
        Build prompt dataset

        Args
        ----
            train_prompts: [Str]
                A list of prompt string.
        Returns
        -------
            torch.utils.data.Dataset
                Dataset with user-defined collate_fn
        """

    def _build_dataloader(self, data, batch_size, dynamic_batch_size_flag=False, is_eval=False):
        """
        build and set the dataloader for the model

        Args:
            data: a list of string
            is_eval: set to `True` to build a dataloader for evaluation (default: `False`)

        :meta private:
        """
        dataset = self.build_dataset(data, is_eval) # pylint: disable=assignment-from-no-return
        consumed_samples = 0
        if not is_eval:
            if self.data_ckpt_manager is not None:
                consumed_samples = self.runtime_args.consumed_samples
        collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None

        dataloader = self.build_dataloader(dataset,
                                           batch_size=batch_size,
                                           collate_fn=collate_fn,
                                           is_eval=is_eval,
                                           dynamic_batch_size_flag=dynamic_batch_size_flag,
                                           consumed_samples=consumed_samples)

        if is_eval:
            self._eval_dataloader = dataloader
            self._eval_data_iter = iter(self._eval_dataloader)
        else:
            self._data_iter = iter(dataloader)
            self._data_iter = cycle(self._data_iter)
            self._dataloader = dataloader

    def build_dataloader(self,
                         dataset,
                         batch_size,
                         collate_fn=None,
                         is_eval=False,
                         dynamic_batch_size_flag=False,
                         consumed_samples=0):
        """
        build the dataloader for the model
        Args:
            dataset: a torch.utils.data.Dataset object
            batch_size: how many samples per batch to load
            collate_fn: set when loading from an map-style dataset (defulat: `None`)
            is_eval: set to `True` to build a dataloader for evaluation (default: `False`)
            consumed_samples: consumed samples

        :meta private:
        """
        log_rank_0(f"Creating DataLoader... consumed_samples: {consumed_samples}", self._logger)
        if is_eval:
            batch_sampler = SingleDataSampler(total_samples=len(dataset),
                consumed_samples=0,
                micro_batch_size=batch_size,
                data_parallel_rank=self.replica_id,
                data_parallel_size=self._num_replica,
                dynamic_batch_size_flag=dynamic_batch_size_flag)
        else:
            batch_sampler = EpisodeDataSampler(total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=batch_size,
                data_parallel_rank=self.replica_id,
                data_parallel_size=self._num_replica,
                sample_per_episode=self.runtime_args.sample_per_episode)
        return DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, pin_memory=True
        )

    def reset_eval_data_iter(self):
        """
        :meta private:
        """
        if self._eval_dataloader is not None:
            self._eval_data_iter = iter(self._eval_dataloader)

    def next_batch(self, is_eval=False):
        """
        :meta private:
        """
        if is_eval:
            return next(self._eval_data_iter)
        else:
            return next(self._data_iter)

    @property
    def num_replica(self):
        """
        :meta private:
        """
        return self._num_replica

    @property
    def num_gpu_per_replica(self):
        """
        :meta private:
        """
        return self._num_gpu_per_replica

    def setup_collective_group(self, rank, world_size, backend, group_name):
        """
        :meta private:
        """
        self._group_names.append(group_name)
        self._world_size = world_size
        col.init_collective_group(
            world_size, rank, backend=backend, group_name=group_name)

    def _destroy_collective_group(self, group_name):
        """
        :meta private:
        """
        from ray.util.collective.collective import _group_mgr # pylint: disable=import-outside-toplevel
        rank = col.get_rank(group_name)
        saved_group: BaseGroup = _group_mgr.get_group_by_name(group_name)
        saved_comm_keys = []
        if isinstance(saved_group, (NCCLGroup, )):
            saved_comm_keys = list(saved_group._dev_comm_map.keys())

        try:
            col.destroy_collective_group(group_name)
        except Exception as e:
            self._logger.warning(f"_destroy_collective_group {group_name} {e}")

        if isinstance(saved_group, (NCCLGroup, )):
            for comm_key in saved_comm_keys:
                group_key = saved_group._generate_group_key(comm_key)
                from ray.util.collective.const import get_store_name # pylint: disable=import-outside-toplevel
                store_name = get_store_name(group_key)
                try:
                    store = ray.get_actor(store_name)
                    if rank == 0:
                        raise RuntimeError(f'{store_name} in group {group_name} should be killed on rank {rank}.')
                    self._logger.debug(f'Kill {store_name} in group {group_name} on rank {rank}')
                    ray.kill(store)
                except ValueError:
                    ...

    def destroy_collective_group(self):
        for group_name in self._group_names:
            self._destroy_collective_group(group_name)
        self._group_names = []

    def get_local_param_ranks(self):
        """
        :meta private:
        """

    def fuse_lora_layer(self):
        """
        :meta private:
        """
        from chatlearn.models.megatron.lora import fuse_lora_layer # pylint: disable=import-outside-toplevel
        fuse_lora_layer(self.model)

    def unfuse_lora_layer(self):
        """
        :meta private:
        """
        from chatlearn.models.megatron.lora import unfuse_lora_layer # pylint: disable=import-outside-toplevel
        unfuse_lora_layer(self.model)

    @property
    def rank(self):
        """
        :meta private:
        """
        return self._rank

    def get_rank(self):
        """
        :meta private:
        """
        return self.rank

    def is_last_rank(self):
        """
        Is last rank.
        """
        return True

    @property
    def parameters(self):
        """
        :meta private:
        """
        if self._parameters is None:
            if not isinstance(self.model, list):
                model = [self.model]
            else:
                model = self.model
            self._parameters = []
            for partition in model:
                for item in partition.parameters():
                    self._parameters.append(item)
        return self._parameters

    @property
    def named_parameters(self):
        """
        :meta private:
        """
        if self._named_parameters is None:
            if not isinstance(self.model, list):
                model = [self.model]
            else:
                model = self.model
            self._named_parameters = {}
            for partition in model:
                for item in partition.named_parameters():
                    self._named_parameters[item[0]] = item[1]
        return self._named_parameters

    @property
    def param_to_name(self):
        """
        :meta private:
        """
        if self._param_to_name is None:
            if not isinstance(self.model, list):
                model = [self.model]
            else:
                model = self.model
            self._param_to_name = {}
            for partition in model:
                for item in partition.named_parameters():
                    self._param_to_name[item[1]] = item[0]
        return self._param_to_name

    @property
    def concat_params_dict(self):
        return self._concat_params_dict

    def get_concat_params_dict(self):
        return self._concat_params_dict

    def set_concat_params_dict(self, _concat_params_dict):
        self._concat_params_dict = _concat_params_dict

    @property
    def to_fix_act_ordering_dict(self):
        return self._to_fix_act_ordering_dict

    def get_to_fix_act_ordering_dict(self):
        return self._to_fix_act_ordering_dict

    def set_to_fix_act_ordering_dict(self, _to_fix_act_ordering_dict):
        self._to_fix_act_ordering_dict = _to_fix_act_ordering_dict

    @property
    def to_fix_qkv_ordering_dict(self):
        return self._to_fix_qkv_ordering_dict

    def get_to_fix_qkv_ordering_dict(self):
        return self._to_fix_qkv_ordering_dict

    def set_to_fix_qkv_ordering_dict(self, _to_fix_qkv_ordering_dict):
        self._to_fix_qkv_ordering_dict = _to_fix_qkv_ordering_dict

    @property
    def to_fix_qkv_ordering_func(self):
        return self._to_fix_qkv_ordering_func

    def get_to_fix_qkv_ordering_func(self):
        return self._to_fix_qkv_ordering_func

    def set_to_fix_qkv_ordering_func(self, _to_fix_qkv_ordering_func):
        self._to_fix_qkv_ordering_func = _to_fix_qkv_ordering_func

    def _set_sync_parameters(self, trainable_param_names, pipe_stage=0, parameters_to_sync=None):
        # pylint: disable=too-many-nested-blocks
        if parameters_to_sync is None:
            parameters_to_sync = defaultdict(list)
        assert pipe_stage not in parameters_to_sync or len(parameters_to_sync[pipe_stage])==0
        concat = []
        set_sync_param_flag = False

        if self.concat_params_dict is not None:
            if isinstance(self.concat_params_dict, dict):
                assert "modules" in self.concat_params_dict
                assert "dim" in self.concat_params_dict
                assert isinstance(self.concat_params_dict["modules"], list)
                concat_modules_list = self.concat_params_dict["modules"]
                concat_dim = self.concat_params_dict["dim"]
            else:
                raise RuntimeError(f"Expect concat_params_dict in {self} to be a dict or None, while {self.concat_params_dict}.")

        if self.to_fix_act_ordering_dict is not None:
            if isinstance(self.to_fix_act_ordering_dict, dict):
                assert "modules" in self.to_fix_act_ordering_dict
                assert "dim" in self.to_fix_act_ordering_dict
                assert isinstance(self.to_fix_act_ordering_dict["modules"], list)
                to_fix_act_ordering_list = self.to_fix_act_ordering_dict["modules"]
                fix_dim = self.to_fix_act_ordering_dict["dim"]
            else:
                raise RuntimeError(f"Expect to_fix_act_ordering_dict in {self} to be a dict or None, while {self.to_fix_act_ordering_dict}.")

        if self.to_fix_qkv_ordering_dict is not None:
            if isinstance(self.to_fix_qkv_ordering_dict, dict):
                assert "modules" in self.to_fix_qkv_ordering_dict
                assert "layer_re" in self.to_fix_qkv_ordering_dict
                assert isinstance(self.to_fix_qkv_ordering_dict["modules"], list)
                to_fix_modules_list = self.to_fix_qkv_ordering_dict["modules"]
                layer_re = self.to_fix_qkv_ordering_dict["layer_re"]
            else:
                raise RuntimeError(f"Expect to_fix_qkv_ordering_dict in {self} to be a dict or None, while {self.to_fix_qkv_ordering_dict}.")

        for name in trainable_param_names:
            if self.concat_params_dict is None and self.to_fix_act_ordering_dict is None:
                set_sync_param_flag = True
                _params_to_sync = self.named_parameters[name]
            else:
                need_concat_or_fix = False
                if self.concat_params_dict is not None:
                    if any([ele in name for ele in concat_modules_list]): # pylint: disable=use-a-generator
                        concat.append(self.named_parameters[name])
                        need_concat_or_fix = True
                        if len(concat) == len(concat_modules_list):
                            set_sync_param_flag = True
                            _params_to_sync = torch.cat(concat, dim=concat_dim)

                if self.to_fix_act_ordering_dict is not None:
                    if any([ele in name for ele in to_fix_act_ordering_list]): # pylint: disable=use-a-generator
                        val = self.named_parameters[name]
                        offset = val.shape[0] // 2
                        w1 = val[:offset,:]
                        w2 = val[offset:,:]
                        need_concat_or_fix = True
                        set_sync_param_flag = True
                        _params_to_sync = torch.cat([w2, w1], dim=fix_dim)

                if not need_concat_or_fix:
                    set_sync_param_flag = True
                    _params_to_sync = self.named_parameters[name]

            if not set_sync_param_flag:
                continue
            if self.to_fix_qkv_ordering_dict is not None:
                from chatlearn.utils.vllm_utils import split_attn_state # pylint: disable=import-outside-toplevel
                m = layer_re.match(name)
                if m is not None:
                    op_name = m.group(2)
                    if op_name in to_fix_modules_list:
                        checkpoint_version = 3.0
                        tp_size = self.module_args.args_dict["tensor_model_parallel_size"]
                        heads = self.module_args.args_dict["num_attention_heads"] // tp_size
                        hidden_size_per_head =  self.module_args.args_dict["hidden_size"] // self.module_args.args_dict["num_attention_heads"]
                        if self._to_fix_qkv_ordering_func is split_attn_state:
                            _num_query_groups = self.module_args.args_dict["num_query_groups"]//tp_size  \
                                if self.module_args.args_dict["group_query_attention"] else heads
                            _params_to_sync = self._to_fix_qkv_ordering_func(
                                _params_to_sync, heads, _num_query_groups, hidden_size_per_head, self.module_args.args_dict["hidden_size"])
                        else:
                            input_shape = _params_to_sync.size()
                            shape = (heads, hidden_size_per_head, 3) + input_shape[1:]
                            division = reduce(operator.mul, shape, 1)
                            num_elements = _params_to_sync.numel()
                            if num_elements == division:
                                # model with gqa dont need to fix qkv ordering.
                                weight_or_bias = m.group(3)
                                _params_to_sync = self._to_fix_qkv_ordering_func(
                                    _params_to_sync, checkpoint_version, 3, heads, hidden_size_per_head
                                )
                                if weight_or_bias == "weight":
                                    _params_to_sync = _params_to_sync.contiguous()
            concat = []
            set_sync_param_flag = False
            parameters_to_sync[pipe_stage].append((name, _params_to_sync))
        return parameters_to_sync

    def set_sync_parameters(self, trainable_param_names, pipe_stage=0, parameters_to_sync=None):
        """
        :meta private:
        """
        if parameters_to_sync is None:
            parameters_to_sync = self._parameters_to_sync
        if pipe_stage not in parameters_to_sync or len(parameters_to_sync[pipe_stage]) == 0:
            self._set_sync_parameters(trainable_param_names, pipe_stage, parameters_to_sync)

    def reset_sync_parameters(self, trainable_param_names, pipe_stage=0):
        self._parameters_to_sync[pipe_stage] = []
        self._set_sync_parameters(trainable_param_names, pipe_stage, self._parameters_to_sync)

    def set_send_parameters(self, trainable_param_names, pipe_stage=0):
        """
        :meta private:
        """
        self._parameters_to_send[pipe_stage] = []
        return self.set_sync_parameters(trainable_param_names, pipe_stage, self._parameters_to_send)

    def set_recv_parameters(self, to_rank, trainable_param_names, pipe_stage=0):
        """
        :meta private:
        """
        parameters_to_recv = defaultdict(list)
        self._parameters_to_recv[to_rank] = parameters_to_recv
        return self.set_sync_parameters(trainable_param_names, pipe_stage, parameters_to_recv)

    def clear_sync_parameters(self):
        self._parameters_to_sync = defaultdict(list)

    def clear_send_recv_parameters(self):
        self._parameters_to_send = defaultdict(list)
        self._parameters_to_recv = defaultdict(list)

    def clear_sync_send_recv_parameters(self):
        self.clear_sync_parameters()
        self.clear_send_recv_parameters()

    def get_parameter_names(self, requires_grad=True):
        """
        :meta private:
        """
        param_to_name = self.param_to_name
        if requires_grad:
            return [param_to_name[param] for param in self.parameters if param.requires_grad]
        else:
            return [param_to_name[param] for param in self.parameters]

    def get_parameter_shape(self, pipe_stage=0, parameters_to_sync=None):
        """
        :meta private:
        """
        if parameters_to_sync is None:
            parameters_to_sync = self._parameters_to_sync
        parameters_shape = []
        for name, param in parameters_to_sync[pipe_stage]:
            parameters_shape.append((name, param.shape))
        return parameters_shape

    def get_parameter(self, name):
        """
        :meta private:
        """
        if name not in self.named_parameters:
            raise Exception(f"parameter {name} not exits")
        return self.named_parameters[name]

    def get_parameter_to_sync(self, name, pipe_stage, to_cpu=False):
        assert pipe_stage in self._parameters_to_sync and len(self._parameters_to_sync[pipe_stage]) > 0
        for name0, param in self._parameters_to_sync[pipe_stage]:
            if name0 == name:
                if to_cpu:
                    param = param.cpu()
                return param

    def get_parameter_to_sync_names(self, pipe_stage):
        return [items[0] for items in self._parameters_to_sync[pipe_stage]]

    def exist_parameter(self, name):
        """
        :meta private:
        """
        return name in self.named_parameters

    def parameter_shape(self, name):
        """
        :meta private:
        """
        return self.get_parameter(name).shape

    def send_recv_parameter(self, name, rank, group_name, func, pipe_stage=0):
        """
        :meta private:
        """
        if self.runtime_args.coalesce_param:
            assert name is None
            tensors = [param.data for _, param in self._parameters_to_sync[pipe_stage]]
            dense_buckets, sparse_bucket = bucket_tensors(tensors, bucket_size_mb=self.runtime_args.coalesced_buffer_mb)
            debug_rank_0(f"{self.name} Got dense_buckets {len(dense_buckets)}, spase_bucket {len(sparse_bucket)}", self._logger)
            for bucket in dense_buckets:
                tensor_changed = func is col.recv
                coalesced_comm_dense(bucket, func, extra_args=(rank, group_name), tensor_changed=tensor_changed)
            for param in sparse_bucket:
                func(param, rank, group_name)
        else:
            tensor = self.get_parameter(name)
            func(tensor, rank, group_name)

    def broadcast_parameter(self, rank, src_rank, group_name, pipe_stage=0):
        """
        :meta private:
        """
        tensors = [param.data for _, param in self._parameters_to_sync[pipe_stage]]
        assert len(tensors) > 0
        dense_buckets, sparse_bucket = bucket_tensors(tensors, bucket_size_mb=self.runtime_args.coalesced_buffer_mb)
        debug_rank_0(f"{self.name} Got dense_buckets {len(dense_buckets)}, spase_bucket {len(sparse_bucket)}", self._logger)
        tensor_changed = rank != src_rank

        for bucket in dense_buckets:
            coalesced_comm_dense(bucket, col.broadcast, extra_args=(src_rank, group_name), tensor_changed=tensor_changed)

        for param in sparse_bucket:
            col.broadcast(param, src_rank, group_name)

    def broadcast_parameter_two_stage(self, to_rank, buffer_rank, rank, src_rank, group_name, pipe_stage=0, stage2=False):
        """
        Arguments:
            to_rank: receive rank in mapping from trainer to inference model.
            buffer_rank: index which tensors of sync buffer to be sended in stage2.
            rank: destination rank in communication group which enumerate receive ranks.
            src_rank: source rank in communication group. always 0.
            group_name: communication group name.
            pipe_stage: pipeline stage. default 0.
            stage2: bool. whether stage2 or not. default False.
        Example: trainer_tp = 4, inference_tp = 8. pipeline_size = 1
            stage1: [(from_rank, to_rank), ...] = [(0, 8), (1, 10), (2, 12), (3, 14)]
            stage2: [(from_rank, to_rank), ...] = [(8, 9), (10, 11), (12, 13), (14, 15)]

            For stage1 pair (0, 8):
                1. call broadcast func: (0 -> 0). src_rank: 0, rank: 0.
                2. call broadcast func: (0 -> 8). src_rank: 0, rank: 1.

                After (0, 8), to_rank 8 received tensor slices of 8 and 9.

            For stage2 pair (8, 9):
                1. call broadcast func: (8 -> 8). src_rank: 0, rank: 0.
                2. call broadcast func: (8 -> 9). src_rank: 0, rank: 1.
                In (8 -> 8), we need to send tp_slice of 'to_rank' 9, so set buffer_rank 9 to fetch tensors in sync buffer.
        """
        tensor_changed = rank != src_rank

        if stage2:
            if tensor_changed:
                parameters_to_sync = self._parameters_to_recv[to_rank]
            else:
                parameters_to_sync = self._parameters_to_send
        else:
            del self._sync_buffer
            self._sync_buffer = defaultdict(list)
            parameters_to_sync = self._parameters_to_sync

        tensors = []
        buffer_num = []
        if stage2 and not tensor_changed and self._sync_buffer:# pylint: disable=too-many-nested-blocks
            idx = 0
            for name, param in parameters_to_sync[pipe_stage]:
                tensors.append(self._sync_buffer[buffer_rank % self.num_mapping][idx])
                buffer_num.append(1)
                idx += 1
            del self._sync_buffer[buffer_rank % self.num_mapping]
        else:
            for name, param in parameters_to_sync[pipe_stage]:
                param_data = param.data
                param_data_shape = param_data.shape
                if rank and self._buffer_num and not stage2:
                    assert name in self._buffer_num, f"{name} in self._buffer_num for rank {rank}"
                    buffer_num.append(self._buffer_num[name])
                elif stage2:
                    buffer_num.append(1)
                else:
                    # Regroup qkv tensors into different tp slices only for inference model which enables vLLM backend.
                    if "attention.query_key_value" in name or \
                            "self_attention.query_key_value" in name or \
                            "self_attention.linear_qkv" in name:
                        tp_size = self.module_args.args_dict["tensor_model_parallel_size"]
                        heads = self.module_args.args_dict["num_attention_heads"] // tp_size
                        hidden_size_per_head = self.module_args.args_dict["hidden_size"] // self.module_args.args_dict["num_attention_heads"]

                        param_shape = (3, heads, hidden_size_per_head) + param_data_shape[1:]
                        division = reduce(operator.mul, param_shape, 1)
                        num_elements = param_data.numel()
                        if num_elements == division:
                            if self.to_fix_qkv_ordering_dict is not None:
                                param_data = param_data.view(param_shape)
                                param_data_list = []
                                head_offset = heads // self._tp_division[name]
                                for idx in range(self._tp_division[name]):
                                    start = idx * head_offset
                                    end = start + head_offset
                                    param_data_list.append(param_data[:,start:end])
                                param_data = torch.concat(param_data_list, dim=0).view(param_data_shape)
                                del param_data_list
                        else:
                            _num_query_groups = self.module_args.args_dict["num_query_groups"]//tp_size  \
                                if self.module_args.args_dict["group_query_attention"] else heads
                            if self.to_fix_qkv_ordering_dict is not None or _num_query_groups == 1:
                                if len(param_data_shape) == 1:
                                    param_data = param.view((heads + 2 * _num_query_groups, hidden_size_per_head))
                                else:
                                    param_data = param.view(
                                        (heads + 2 * _num_query_groups, hidden_size_per_head, self.module_args.args_dict["hidden_size"]))
                                param_data_list = []
                                head_offset = heads // self._tp_division[name]
                                for idx in range(self._tp_division[name]):
                                    q_start = idx * head_offset
                                    q_end = q_start + head_offset
                                    k_start = (heads + idx) if _num_query_groups // self._tp_division[name] else heads
                                    k_end = k_start + 1
                                    v_start = k_start + _num_query_groups
                                    v_end = v_start + 1

                                    q_proj = param_data[q_start:q_end].contiguous()
                                    k_proj = param_data[k_start:k_end].contiguous()
                                    v_proj = param_data[v_start:v_end].contiguous()

                                    qkv_proj = torch.cat([q_proj, k_proj, v_proj], dim=0)

                                    if len(param_data_shape) == 1:
                                        qkv_proj = qkv_proj.reshape(-1).contiguous()
                                    else:
                                        qkv_proj = qkv_proj.reshape(-1, self.module_args.args_dict["hidden_size"]).contiguous()

                                    param_data_list.append(qkv_proj)
                                param_data = torch.concat(param_data_list, dim=0)
                                del param_data_list

                    # Regroup these tensors into different tp slices.
                    # Output: [tp_slice_0, tp_slice_1, ...]
                    # Comment:
                    #   src -> dst: [w, h * tp_size] -> tp_size * [w, h]
                    #       'self_attention.dense' in QWen and LLama2 legacy
                    #       'mlp.dense_4h_to_h' in QWen and LLama2 legacy model
                    #       'mlp.linear_fc2' in LLama2 mcore model
                    #       'mlp.shared_experts.dense_4h_to_h in QWen-MoE model
                    #   src -> dst: [w * tp_size, h] -> tp_size * [w, h]
                    #       'mlp.dense_h_to_4h' in QWen and LLama2 legacy
                    #       'mlp.linear_fc1' in LLama2 mcore model
                    #       'mlp.w1' in QWen model only for vLLM backend
                    #       'mlp.shared_experts.dense_h_to_4h in QWen-MoE model
                    if (
                        "self_attention.dense" in name
                        or "mlp.dense_4h_to_h" in name
                        or "mlp.linear_fc2" in name
                        or "mlp.shared_experts.dense_4h_to_h" in name
                    ):
                        param_data_list = []
                        col_offset = param_data_shape[1] // self._tp_division[name]
                        for idx in range(self._tp_division[name]):
                            start = idx * col_offset
                            end =  start + col_offset
                            param_data_list.append(param_data[:,start:end])
                        param_data = torch.concat(param_data_list, dim=0).view(param_data_shape)
                        del param_data_list
                    if ("mlp.dense_h_to_4h" in name
                        or "mlp.linear_fc1" in name
                        or ("mlp.w1" in name and self.concat_params_dict is not None)
                        or "mlp.shared_experts.dense_h_to_4h" in name):
                        param_data_list = []
                        row_offset = param_data_shape[0] // self._tp_division[name] // 2
                        for idx in range(self._tp_division[name]):
                            w1_start = idx * row_offset
                            w1_end = w1_start + row_offset
                            w2_start = (idx + self._tp_division[name]) * row_offset
                            w2_end = w2_start + row_offset
                            param_data_list.append(
                                torch.concat([param_data[w1_start:w1_end,:], param_data[w2_start:w2_end,:]], dim=0))
                        param_data = torch.concat(param_data_list, dim=0).view(param_data_shape)
                        del param_data_list
                    buffer_num.append(1)
                tensors.append(param_data)

        assert len(tensors) > 0
        dense_buckets, sparse_bucket = bucket_tensors_two_stage(
            tensors, bucket_size_mb=self.runtime_args.coalesced_buffer_mb,
            buffer_num=None if stage2 else buffer_num, tensor_changed=tensor_changed and not stage2)
        debug_rank_0(f"{self.name} Got dense_buckets {len(dense_buckets)}, sparse_bucket {len(sparse_bucket)}", self._logger)

        for bucket in dense_buckets:
            index = 0 if stage2 else (to_rank % self.num_mapping)
            all_buffers = coalesced_comm_dense_two_stage(
                bucket, col.broadcast, rank,
                extra_args=(src_rank, group_name), tensor_changed=tensor_changed,
                stage2=stage2, index=index)
            if tensor_changed and not stage2:
                for key, value in all_buffers.items():
                    self._sync_buffer[key] += value

        for param in sparse_bucket:
            col.broadcast(param, src_rank, group_name)

        self.empty_cache()

    def send_parameter(self, name, dst_rank, group_name, pipe_stage=0):
        """
        :meta private:
        """
        self.send_recv_parameter(name, dst_rank, group_name, col.send, pipe_stage)

    def recv_parameter(self, name, src_rank, group_name, pipe_stage=0):
        """
        :meta private:
        """
        self.send_recv_parameter(name, src_rank, group_name, col.recv, pipe_stage)

    def ray_put_parameter(self, name, group_name, pipe_stage=0):
        """
        :meta private:
        """
        name2ref = {}
        if self.runtime_args.coalesce_param:
            assert name is None
            tensors = [param.data for _, param in self._parameters_to_sync[pipe_stage]]
            dense_buckets, sparse_bucket = bucket_tensors(tensors, bucket_size_mb=self.runtime_args.coalesced_buffer_mb)
            debug_rank_0(f"{self.name} Put dense_buckets {len(dense_buckets)}, spase_bucket {len(sparse_bucket)}", self._logger)
            for bucket_id, bucket in enumerate(dense_buckets):
                flat_tensors = _flatten_dense_tensors(bucket)
                flat_tensors_ref = ray.put(flat_tensors)
                name2ref[group_name + ":dense_bucket_" + str(bucket_id)] = flat_tensors_ref
            for param_id, param in enumerate(sparse_bucket):
                param_ref = ray.put(param)
                name2ref[group_name + ":sparse_bucket_" + str(param_id)] = param_ref
        else:
            tensor = self.get_parameter(name)
            tensor_ref = ray.put(tensor)
            name2ref[group_name + ":" + name] = tensor_ref
        return name2ref

    def ray_get_parameter(self, name, group_name, name2ref, pipe_stage=0):
        """
        :meta private:
        """
        if self.runtime_args.coalesce_param:
            assert name is None
            tensors = [param.data for _, param in self._parameters_to_sync[pipe_stage]]
            dense_buckets, sparse_bucket = bucket_tensors(tensors, bucket_size_mb=self.runtime_args.coalesced_buffer_mb)
            debug_rank_0(f"{self.name} Get dense_buckets {len(dense_buckets)}, spase_bucket {len(sparse_bucket)}", self._logger)
            for bucket_id, bucket in enumerate(dense_buckets):
                put_ref = name2ref[group_name + ":dense_bucket_" + str(bucket_id)]
                flat_tensors = ray.get(put_ref)
                for tensor, synced in zip(
                    bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
                    tensor.copy_(synced)
            for param_id, param in enumerate(sparse_bucket):
                put_ref = name2ref[group_name + ":sparse_bucket_" + str(param_id)]
                param.copy_(ray.get(put_ref))
        else:
            tensor = self.get_parameter(name)
            put_ref = name2ref[group_name + ":" + name]
            tensor.copy_(ray.get(put_ref))

    def pipeline_model_parallel_size(self):
        """
        :meta private:
        """
        return self.module_args.pipeline_model_parallel_size

    def tensor_model_parallel_size(self):
        """
        :meta private:
        """
        return self.module_args.tensor_model_parallel_size

    def expert_model_parallel_size(self):
        """
        :meta private:
        """
        return self.module_args.expert_model_parallel_size

    def num_layers(self):
        """
        :meta private:
        """

    def set_storage(self, storage):
        """
        :meta private:
        """
        self._storage = storage

    def timers(self, name):
        """
        :meta private:
        """
        if self._timers is None:
            self._timers = Timers()
        return self._timers(name)

    def timer_summary(self, e2e_cost=None):
        """
        :meta private:
        """
        if self._timers:
            return self._timers.log(e2e_cost=e2e_cost)

    def add_padding_config(self, key, padding_value=0.0, padding_type="right"):
        """
        Add spectial padding config for certain value.

        Args
        ----
        key: str
            The key for data to be padded.
        padding_value: float
            Padding value, default is 0.
        padding_type: str
            Default right, can be right/left.
        """
        self._padding_config[key] = {"padding_value": padding_value, "padding_type": padding_type}

    def padding_config(self):
        """
        :meta private:
        """
        return self._padding_config

    def peak_memory(self):
        """
        :meta private:
        """
        return 0.0

    @property
    def resume_training(self):
        """
        resume training from last checkpoint.
        """
        return self._resume_training

    def get_address(self):
        """
        Get node address

        :meta private:
        """
        return self._address

    def is_master_node(self):
        """
        Whether this node is master node.
        :meta private:
        """
        return self._is_master_node

    def set_src_parameter_model(self, src_model):
        """
        src_model that sync parameter to current model
        :meta private:
        """
        self._src_parameter_model = src_model

    @property
    def src_parameter_model(self):
        """
        src_model that sync parameter to current model
        """
        return self._src_parameter_model

    def offload_optimizer_states(self):
        """
        offload optimizer states
        """

    def onload_optimizer_states(self):
        """
        onload optimizer states
        """

    def offload_main_weights(self):
        """
        offload main weights
        """

    def onload_main_weights(self):
        """
        onload main weights
        """

    def offload_weights(self):
        """
        offload weights
        """

    def onload_weights(self):
        """
        onload weights
        """

    def free_grad_buffers(self):
        """
        free grad buffers and related tensors
        """

    def build_grad_buffers(self):
        """
        build grad buffers and related tensors
        """

    def onload(self):
        pass

    def offload(self):
        pass

    @property
    def world_size(self):
        pass

    @property
    def data_parallel_size(self):
        """
        data parallel size

        :meta private:
        """

    @property
    def data_parallel_rank(self):
        """
        data parallel rank

        :meta private:
        """

    def empty_cache(self):
        """
        :meta private:
        """

    def get_data_parallel_rank(self):
        return self.data_parallel_rank

    def get_data_parallel_size(self):
        return self.data_parallel_size

    def get_pipeline_stage_layer_num(self):
        pass

    def get_pipeline_stage_layer_offset(self):
        return 0
