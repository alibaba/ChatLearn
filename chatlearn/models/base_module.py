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
from typing import Dict, TYPE_CHECKING
from itertools import cycle
from pathlib import Path
import math
import os
import torch
import ray

from chatlearn.data.sampler import MultiDatasetSampler
from chatlearn.data.data import RLHFDataLoader
from chatlearn.checkpoint.checkpoint_manager import CheckpointManager
from chatlearn.utils import future
from chatlearn.utils.constant import LOG_START
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.global_vars import set_global_variables
from chatlearn.utils.logger import logger
from chatlearn.utils.logger import log_rank_0, setup_logger
from chatlearn.utils.timer import Timers
from chatlearn.utils.constant import REF_LIST, INDEX_TAG
from chatlearn.utils.utils import get_host_addr, map_reduce_metrics, slice_data_list_by_index
from chatlearn.launcher import dlc_utils
from chatlearn.configs.base import BaseModelConfig
from chatlearn.synchronizer import name_to_mapper_cls, GeneralCommunicator

if TYPE_CHECKING:
    from chatlearn.synchronizer.structs import BucketInfo


class BaseModule:
    """The base class for all chatlearn models."""
    def __init__(self, name: str, args=None, replica_id: int=0):
        """The base class for all chatlearn models. After setup, the initialized
        base module on the remote actor can be used for training/inferencing.

        Args:
            name (str): The name of this module
            args (Any, optional): The arguments. Defaults to None.
            replica_id (int, optional): The replica id of this module. Defaults to 0.
        """
        logger.info(f"{LOG_START} basemodule {name} init start")
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
        self._is_colocate = False

        self._num_gpu_per_replica = self.total_gpu // self.module_args.num_replica
        self._num_replica = self.module_args.num_replica

        self._param_ranks = None
        self._named_parameters = None
        self._param_to_name = None
        self._parameters = None
        self._coalesced_parameters = None
        self.error_signal = None


        self._dataloader = None
        self._eval_dataloader = None
        self._timers = Timers()
        self._data_iter = None
        self._eval_data_iter = None
        self.call_funcs = []
        self.trainable_funcs = []
        self._data_ckpt_manager = None
        self._peak_memory = 0

        # current compute iteration
        self._iteration = 0
        self._train_iteration = 0
        self._episode_id = 0
        self._finalized = False
        self._resume_training = False
        self._address = dlc_utils.get_addr() if dlc_utils.in_dlc_env() else get_host_addr()
        self._is_master_node = os.environ.get("RANK", '0') == '0'
        self._logger = setup_logger(model_name=self.name, ip_addr=self._address)
        self.profiler = None
        self.synchronizer = None
        self._metric_prefix = ""
        self._metric_list = []
        self._stage_resume_done = False
        logger.info(f"{LOG_START} basemodule {name} init done")

    @property
    def is_colocate(self):
        return self._is_colocate

    def set_colocate(self, flag):
        self._is_colocate = flag

    def get_runtime_args(self):
        return self.runtime_args

    @property
    def runtime_args(self):
        """
        Return the arguments related to alignment training,
        the settings that are specified under the "runtime" section of the YAML configuration file.
        """
        return self._runtime_args

    @property
    def module_args(self):
        """
        Return module arguments. module_args include `num_gpu`, `gpu_per_process`, `model_config_file`, etc.
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
        if self.runtime_args.data_checkpoint_path is not None:
            self._data_ckpt_manager = CheckpointManager(self, self.runtime_args.data_checkpoint_path,
                                                       self.runtime_args.max_data_ckpt_nums,
                                                       self.runtime_args.load_data_checkpoint_iteration)
            if self.runtime_args.enable_resume_training:
                meta = self._data_ckpt_manager.resume()
                if meta:
                    self._resume_training = self.runtime_args.consumed_samples > 0
                    start_episode = meta["episode"] + 1
                    self._episode_id = start_episode
                    self._iteration = start_episode * math.ceil(self.runtime_args.sample_per_episode / \
                        self._num_replica / self.module_args.generation_batch_size)

                    log_rank_0(
                        f"{self.name} resume training {self._resume_training}: "
                        f"set start iteration to {self._iteration} and episode id to {self._episode_id}",
                        self._logger)
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

    def train_step(self, data_list, **kwargs):
        """
        Perform train_step for one batch, including a list of micro-batches.

        Args
        ----
        data : [Dict]
            A list of micro-batch for train_step, type of each micro-batch is dict
        iteration : int
            local train iteration
        """

    def _preprocess_impl(self, data):
        # Preprocess after get list of sample
        return data

    def data_fetch(self, data_ref, train_func: bool):
        # Get data from remote dataset
        data_list = future.get(data_ref)
        # For data in trainer, data_list is [microbatch0, microbatch1, ...]
        # For data in environment which is inter-node in graph, data_list is [inque_input_node0, inque_input_node1, ...]
        if not train_func:
            batched_data_list = [[] for _ in range(len(data_list))]
            for idx, data_obj in enumerate(data_list):
                if isinstance(data_obj, list):
                    batched_data_list[idx] = data_obj
                if REF_LIST in data_obj:
                    for data_slice in data_obj[REF_LIST]:
                        batched_data_list[idx].extend(data_slice)
                if INDEX_TAG in data_obj:
                    batched_data_list[idx] = slice_data_list_by_index(batched_data_list[idx], data_obj[INDEX_TAG])
            if len(batched_data_list) > 1:
                # When current node have several input nodes, we need to merge them
                # Data size for each input node must be same
                assert len({len(input_list) for input_list in batched_data_list}) == 1
                data_list = [{k: v for d in group for k, v in d.items()} for group in zip(*batched_data_list)]
            else:
                data_list = batched_data_list[0]
        else:
            data_list = data_list[0]
        return self._preprocess_impl(data_list)

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
        self._episode_id += 1

    def build_dataset(self, prompts, is_eval=False):
        """
        Build prompt dataset

        Args
        ----
            prompts: [Str]
                A list of prompt string.
        Returns
        -------
            torch.utils.data.Dataset
                Dataset with user-defined collate_fn
        """

    def build_all_dataset(self, prompts_list, is_eval=False):
        """
        Build all prompt datasets

        Args
        ----
            prompts_list: List[List[Str]]
                A list of prompt string lists.
        Returns
        -------
            List[torch.utils.data.Dataset]
                A list of Dataset with user-defined collate_fn
        """
        all_datasets = []
        for prompts in prompts_list:
            all_datasets.append(
                self.build_dataset(prompts, is_eval)
            )
        return all_datasets

    def _build_dataloader(self, data, sample_per_episode, is_eval=False):
        """
        build and set the dataloader for the model

        Args:
            data: a list of string
            is_eval: set to `True` to build a dataloader for evaluation (default: `False`)

        :meta private:
        """
        all_datasets = self.build_all_dataset(data, is_eval) # pylint: disable=assignment-from-no-return
        consumed_samples = 0
        data_ratio = self.runtime_args.data_ratio
        shuffle = self.runtime_args.data_shuffle
        data_rerank = self.runtime_args.data_rerank
        if not is_eval:
            if self.data_ckpt_manager is not None:
                consumed_samples = self.runtime_args.consumed_samples
        collate_fn = all_datasets[0].collate_fn if hasattr(all_datasets[0], 'collate_fn') else None
        drop_last = self.module_args['drop_last'] if 'drop_last' in self.module_args else False
        dataloader = self.build_dataloader(all_datasets,
                                           sample_per_episode=sample_per_episode,
                                           collate_fn=collate_fn,
                                           is_eval=is_eval,
                                           consumed_samples=consumed_samples,
                                           data_ratio=data_ratio,
                                           shuffle=shuffle,
                                           drop_last=drop_last,
                                           data_rerank=data_rerank)

        if is_eval:
            self._eval_dataloader = dataloader
            self._eval_data_iter = iter(self._eval_dataloader)
        else:
            self._data_iter = iter(dataloader)
            self._data_iter = cycle(self._data_iter)
            self._dataloader = dataloader

    def build_dataloader(self,
                         all_datasets,
                         sample_per_episode,
                         collate_fn=None,
                         is_eval=False,
                         consumed_samples=0,
                         data_ratio=None,
                         shuffle=True,
                         drop_last=False,
                         data_rerank=True):
        """
        build the dataloader for the model
        Args:
            all_datasets: a list of torch.utils.data.Dataset objects
            batch_size: how many samples per batch to load
            collate_fn: set when loading from an map-style dataset (defulat: `None`)
            is_eval: set to `True` to build a dataloader for evaluation (default: `False`)
            consumed_samples: consumed samples (default: `0`)
            data_ratio: ratio of samples for each dataset (default: `None`)
            drop_last: whether to drop last samples (default: `False`)

        :meta private:
        """
        log_rank_0(
            f"Creating DataLoader... consumed_samples: {consumed_samples}, "
            f"data_ratio: {data_ratio}",
            self._logger
        )
        if "num_inference_per_prompt" in self.module_args:
            num_inference_per_prompt = self.module_args["num_inference_per_prompt"]
        else:
            num_inference_per_prompt = 1
        self._logger.info(f"====Data Rerank: {data_rerank}")
        if is_eval:
            batch_sampler = MultiDatasetSampler(
                dataset_sizes=[len(dataset) for dataset in all_datasets],
                sample_per_episode=sample_per_episode,
                shuffle=False,
                is_eval=True,
                data_parallel_rank=self.replica_id,
                data_parallel_size=self._num_replica
            )
        else:
            batch_sampler = MultiDatasetSampler(
                dataset_sizes=[len(dataset) for dataset in all_datasets],
                sample_per_episode=sample_per_episode,
                data_ratio=data_ratio,
                consumed_samples=consumed_samples,
                num_inference_per_prompt=num_inference_per_prompt,
                shuffle=shuffle,
                is_eval=False,
                data_parallel_rank=self.replica_id,
                data_parallel_size=self._num_replica,
                drop_last="drop" if drop_last else "cycle",
                data_rerank=data_rerank
            )
        return RLHFDataLoader(
            all_datasets,
            batch_sampler,
            collate_fn=collate_fn,
            data_parallel_rank=self.replica_id,
            data_parallel_size=self._num_replica,
            num_inference_per_prompt=num_inference_per_prompt
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

    def timers(self, name):
        return self._timers(name)

    def timer_summary(self, e2e_cost=None):
        return self._timers.log(return_dict=True, e2e_cost=e2e_cost)

    def get_and_clear_metrics(self):
        """
        get logging metrics
        """
        if self._metric_list is None or len(self._metric_list) == 0:
            return self._metric_prefix, {}

        reduced_metrics = map_reduce_metrics(self._metric_list)
        self._metric_list = []
        return self._metric_prefix, reduced_metrics

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

    def empty_cache(self):
        """
        :meta private:
        """

    # TODO: remove the below two APIs in the future
    def get_data_parallel_rank(self):
        return self.data_parallel_rank

    def get_data_parallel_size(self):
        return self.data_parallel_size

    @property
    def data_parallel_rank(self):
        """
        data parallel rank

        :meta private:
        """
    @property
    def data_parallel_size(self):
        """
        data parallel size

        :meta private:
        """

    def enable_stage_resume(self, is_eval):
        """
        check whether to resume stage outputs.
        """
        if is_eval:
            return False
        if self.module_args.get("enable_stage_resume", False):
            assert self.runtime_args.data_checkpoint_path, \
                "data_checkpoint_path must be set for stage resume."
            return True
        return False

    def get_stage_outputs_path(self, iteration):
        """
        get path for stage outputs.
        """
        save_dir = self.runtime_args.data_checkpoint_path
        save_path = f"{save_dir}/{iteration}/{self.name}_replica_{self.replica_id}.pt"
        save_path_meta = f"{save_dir}/{iteration}/{self.name}_replica_{self.replica_id}_meta.txt"
        return save_path, save_path_meta

    def load_stage_outputs(self, is_eval, iteration):
        """
        load stage outputs for resume.
        """
        outputs = None
        # only load once for each launching.
        if self.enable_stage_resume(is_eval) and not self._stage_resume_done:
            self._stage_resume_done = True
            save_path, save_path_meta=self.get_stage_outputs_path(iteration)
            if os.path.exists(save_path) and os.path.exists(save_path_meta):
                try:
                    with open(save_path_meta, "r", encoding='utf-8') as f:
                        replica_id = int(f.readline())
                    if replica_id == self.replica_id:
                        outputs = torch.load(save_path)
                        logger.info(f"resume stage outputs for model:{self.name}, path:{save_path}")
                except ValueError:
                    logger.warning(f"ignore incomplete stage outputs, path:{save_path}")
        return outputs

    def save_stage_outputs(self, is_eval, outputs, iteration):
        """
        save stage outputs for resume.
        """
        if self.enable_stage_resume(is_eval):
            save_path, save_path_meta=self.get_stage_outputs_path(iteration)
            logger.info(f"Start to save stage outputs:{save_path}")
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(outputs, save_path)
            # save meta
            with open(save_path_meta, "w", encoding='utf-8') as f:
                f.write(f"{self.replica_id}")
            logger.info(f"Finished to save stage outputs:{save_path}")

    # NOTE: the following APIs are for updated parameter synchronization.
    def set_mapper(self, mapper_name: str, dst_model_config: BaseModelConfig):
        self.mapper = name_to_mapper_cls(mapper_name)(
            dst_model_config=dst_model_config,
            model=self
        )

    def generate_sync_mapping(self, dst_name_to_metadata):
        return self.mapper.generate_sync_mapping(dst_name_to_metadata)

    def set_param_ids(self, global_name_to_param_id: Dict[str, int]):
        self.local_name_to_param_id = {
            v: global_name_to_param_id[k]
            for k, v in self.global_name_to_local_name.items()
        }
        self.param_id_to_local_name = {
            global_name_to_param_id[k]: v
            for k, v in self.global_name_to_local_name.items()
        }

    def parameter_sync(self):
        """Perform parameter synchronization on this worker."""
        if self.synchronizer is None:
            raise ValueError("Synchronizer is not initialized.")
        return self.synchronizer.parameter_sync()

    def post_parameter_sync(self):
        """Release resources after parameter synchronization."""

    def get_gpu_info(self):
        """return a unique string to identify the GPU"""
        node_id = ray.get_runtime_context().get_node_id()
        gpu_ids = ray.get_gpu_ids()
        assert len(gpu_ids) == 1, "Not Supported"
        return f"{node_id}-{gpu_ids[0]}"

    def set_synchronizer(
        self,
        synchronizer_name: str='general',
        **kwargs
    ):
        """initialize the synchronizer on this rank.

        Args:
            synchronizer_name (str): type name of the synchronizer
            kwargs (Dict): kwargs for the synchronizer
        """
        if synchronizer_name != "general":
            raise ValueError(f"Unrecognized Synchronizer {synchronizer_name}")
        self.synchronizer = GeneralCommunicator(model=self, **kwargs)

    def call_synchronizer_func(self, func_name, *args, **kwargs):
        """Call some apis of sychronizers"""
        return getattr(self.synchronizer, func_name)(*args, **kwargs)

    def get_mem_info(self):
        return torch.cuda.mem_get_info()
