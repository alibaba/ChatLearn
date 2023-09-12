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
"""RLHF base module"""

from collections import defaultdict
from itertools import cycle

import ray
import ray.util.collective as col
from torch.utils.data import DataLoader
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from chatlearn.data.data import EpisodeDataLoader
from chatlearn.checkpoint.checkpoint_manager import CheckpointManager
from chatlearn.utils import future
from chatlearn.utils.dist_utils import bucket_tensors, coalesced_comm_dense
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.global_vars import set_global_variables
from chatlearn.utils.logger import log_rank_0, debug_rank_0
from chatlearn.utils.timer import Timers


class RLHFModule:
    """RLHFModule is the base class for RLHF models.

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
        self.total_device = args.num_device
        self.gpu_per_process = args.gpu_per_process
        self.trainable = args.trainable
        self._rlhf_args = self.global_args.rlhf_args
        self._module_args = args
        self.global_args.active_module_args = self._module_args
        self.replica_id = replica_id
        self.config_dir = args.config_dir
        self._num_device_per_replica = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
        assert self._num_device_per_replica <= self.total_device
        assert self.total_device % self._num_device_per_replica == 0
        if not self.trainable:
            self._num_replica = args.num_device // self._num_device_per_replica
        else:
            # For trainable models, perform the DP inside DistActor
            self._num_replica = 1
            self._num_device_per_replica = self.total_device
        assert self._num_replica >= 1
        self._param_ranks = None
        self._named_parameters = None
        self._param_to_name = None
        self._parameters = None
        self._coalesced_parameters = None
        self.error_signal = None
        self._rank = None
        self._world_size = None
        self._group_name = None
        self._dataloader = None
        self._eval_dataloader = None
        self._kl_coef = None
        self._padding_config = {}
        self._storage = None
        self._timers = None
        self._data_iter = None
        self._eval_data_iter = None
        self.call_funcs = []
        self.data_ckpt_manager = None
        self._peak_memory = 0
        self._return_rlhf_data = self._module_args.return_rlhf_data
        self._parameters_to_sync = defaultdict(list)
        # current compute iteration
        self._iteration = 0
        self.enable_lora = self._module_args.lora.enable_lora
        self._eval_func_name = None
        self._finalized = False

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
    def rlhf_args(self):
        """
        Return the arguments related to RLHF training,
        the settings that are specified under the "rlhf" section of the YAML configuration file.
        """
        return self._rlhf_args

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
        Return module arguments. module_args include `num_device`, `gpu_per_process`, `model_config_file`, etc.
        """
        return self._module_args

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

    def model_setup(self):
        """
        :meta private:
        """
        self.setup()

    def forward_step(self, data, iteration=None):
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

    def train_step(self, data, train_info):
        """
        Perform train_step for one batch, including a list of micro-batches.

        Args
        ----
        data : [Dict]
            A list of micro-batch for train_step, type of each micro-batch is dict
        train_info : Dict
            A dict of training meta, includes training information, e.g., "iteration"
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

    def save_data_checkpoint(self, replica_id, iteration, ppo_iter):
        """
        Save checkpoint for dataloader.

        :meta private:
        """
        if self.data_ckpt_manager is not None:
            self.data_ckpt_manager.save_checkpoint(replica_id, iteration, ppo_iter)

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
        return "ok"

    def before_episode(self):
        """
        Operations before one episode.
        """

    def after_episode(self):
        """
        Operations after one episode.
        """

    def build_dataset(self, train_prompts):
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

    def _build_dataloader(self, data, sample_per_episode_per_replica=-1, is_eval=False):
        """
        build and set the dataloader for the model

        Args:
            data: a list of string
            sample_per_episode_per_replica: an integer indicate how many samples 
                per episode and per replica will consume (default: `-1`)
            is_eval: set to `True` to build a dataloader for evaluation (default: `False`)

        :meta private:
        """
        if not is_eval:
            assert sample_per_episode_per_replica > 0, \
                "The dataloader for training expect positive sample_per_episode_per_replica, "\
                f"but got {sample_per_episode_per_replica}"
        dataset = self.build_dataset(data) # pylint: disable=assignment-from-no-return
        assert hasattr(dataset, 'collate_fn'), \
            f"{dataset.__class__.__name__} has no attribute `collate_fn`. If you would like "\
            "to use the default collate_fn to batch samples, try adding `self.collate_fn = None` "\
            "to your Dataset object"
        dataloader = self.build_dataloader(dataset,
                                           batch_size=self.module_args.generation_batch_size,
                                           shuffle=False,
                                           collate_fn=dataset.collate_fn,
                                           sample_per_episode_per_replica=sample_per_episode_per_replica,
                                           is_eval=is_eval)

        if is_eval:
            self._eval_dataloader = dataloader
            self._eval_data_iter = iter(self._eval_dataloader)
        else:
            if self.data_ckpt_manager is None and self.rlhf_args.data_checkpoint_path is not None:
                self.data_ckpt_manager = CheckpointManager(self, self.rlhf_args.data_checkpoint_path,
                                                           self.rlhf_args.max_data_ckpt_nums,
                                                           self.rlhf_args.load_data_checkpoint_iteration)
                self.data_ckpt_manager.resume()
            if self.data_ckpt_manager is not None:
                dataloader = self.data_ckpt_manager.data_loader(dataloader, is_cycle=True)
                self._data_iter = iter(dataloader)
            else:
                self._data_iter = iter(dataloader)
                self._data_iter = cycle(self._data_iter)
            self._dataloader = dataloader

    def build_dataloader(self,
                         dataset,
                         batch_size,
                         shuffle=False,
                         collate_fn=None,
                         sample_per_episode_per_replica=-1,
                         is_eval=False):
        """
        build the dataloader for the model

        Args:
            dataset: a torch.utils.data.Dataset object
            batch_size: how many samples per batch to load
            shuffle: set to `True` to shuffle dataset
                at every epoch (default: `False`)
            collate_fn: set when loading from an map-style dataset (defulat: `None`)
            sample_per_episode_per_replica: an integer indicate how many samples 
                per episode and per replica will consume (default: `-1`)
            is_eval: set to `True` to build a dataloader for evaluation (default: `False`)

        :meta private:
        """
        if self.rlhf_args.enable_indivisible_batch_size and not is_eval:
            log_rank_0("Creating EpisodeDataLoader...")
            assert sample_per_episode_per_replica > 0, \
                "sampler_per_episode_per_replica must be an integer greater than 0 "\
                f"when using `EpisodeDataLoader`, but got {sample_per_episode_per_replica}."
            return EpisodeDataLoader(
                dataset, sample_per_episode_per_replica, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
            )
        else:
            log_rank_0("Creating DataLoader...")
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
            )

    def reset_eval_data_iter(self):
        """
        :meta private:
        """
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
    def num_device_per_replica(self):
        """
        :meta private:
        """
        return self._num_device_per_replica

    def setup_collective_group(self, rank, world_size, backend, group_name):
        """
        :meta private:
        """
        self._group_name = group_name
        self._world_size = world_size
        col.init_collective_group(
            world_size, rank, backend=backend, group_name=group_name)

    def destroy_collective_group(self):
        """
        :meta private:
        """
        col.destroy_collective_group(self._group_name)

    def set_param_ranks(self, param_ranks):
        """
        Set the ranks for parameters of first replica.
        """
        self._param_ranks = param_ranks

    def get_param_ranks(self):
        """
        :meta private:
        """

    def fuse_lora_layer(self):
        """
        :meta private:
        """
        from chatlearn.opt.lora import fuse_lora_layer # pylint: disable=import-outside-toplevel
        fuse_lora_layer(self.model)

    def unfuse_lora_layer(self):
        """
        :meta private:
        """
        from chatlearn.opt.lora import unfuse_lora_layer # pylint: disable=import-outside-toplevel
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

    def set_sync_parameters(self, trainable_param_names, pipe_stage=0):
        """
        :meta private:
        """
        if pipe_stage not in self._parameters_to_sync or len(self._parameters_to_sync[pipe_stage]) == 0:
            for name in trainable_param_names:
                self._parameters_to_sync[pipe_stage].append(self.named_parameters[name])

    def get_parameter_names(self, requires_grad=True):
        """
        :meta private:
        """
        param_to_name = self.param_to_name
        if requires_grad:
            return [param_to_name[param] for param in self.parameters if param.requires_grad]
        else:
            return [param_to_name[param] for param in self.parameters]

    def get_parameter(self, name):
        """
        :meta private:
        """
        if name not in self.named_parameters:
            raise Exception(f"parameter {name} not exits")
        return self.named_parameters[name]

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
        if self.rlhf_args.coalesce_param:
            assert name is None
            tensors = [param.data for param in self._parameters_to_sync[pipe_stage]]
            dense_buckets, sparse_bucket = bucket_tensors(tensors, bucket_size_mb=self.rlhf_args.coalesced_buffer_mb)
            debug_rank_0(f"{self.name} Got dense_buckets {len(dense_buckets)}, spase_bucket {len(sparse_bucket)}")
            for bucket in dense_buckets:
                tensor_changed = func is col.recv
                coalesced_comm_dense(bucket, func, extra_args=(rank, group_name), tensor_changed=tensor_changed)
            for param in sparse_bucket:
                func(param, rank, group_name)
        else:
            tensor = self.get_parameter(name)
            func(tensor, rank, group_name)

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
        if self.rlhf_args.coalesce_param:
            assert name is None
            tensors = [param.data for param in self._parameters_to_sync[pipe_stage]]
            dense_buckets, sparse_bucket = bucket_tensors(tensors, bucket_size_mb=self.rlhf_args.coalesced_buffer_mb)
            debug_rank_0(f"{self.name} Put dense_buckets {len(dense_buckets)}, spase_bucket {len(sparse_bucket)}")
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
        if self.rlhf_args.coalesce_param:
            assert name is None
            tensors = [param.data for param in self._parameters_to_sync[pipe_stage]]
            dense_buckets, sparse_bucket = bucket_tensors(tensors, bucket_size_mb=self.rlhf_args.coalesced_buffer_mb)
            debug_rank_0(f"{self.name} Get dense_buckets {len(dense_buckets)}, spase_bucket {len(sparse_bucket)}")
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

    def tensor_model_parallel_size(self):
        """
        :meta private:
        """

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

    def timer_summary(self):
        """
        :meta private:
        """
        if self._timers:
            return self._timers.log()

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

    def register_func(self, name):
        """
        register func to be called by engine

        :meta private:
        """
        self._assert_not_finalized()
        self.call_funcs.append(name)

    def register_eval_func(self, name='eval_step'):
        """
        Register func to be called by eval engine

        Args
        ----
            name: str
                function name
        """
        self._assert_not_finalized()
        self.register_func(name)
        self._eval_func_name = name

    @property
    def eval_func_name(self):
        """
        :meta private:
        """
        return self._eval_func_name

    def add_step(self, step):
        """
        :meta private:
        """
        if self.data_ckpt_manager is not None:
            self.data_ckpt_manager.add_step(step)

    def set_start_iteration(self, start_iteration):
        """
        :meta private:
        """
        self._iteration = start_iteration
