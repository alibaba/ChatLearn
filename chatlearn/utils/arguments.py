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
"""arguments from command or yaml."""

import argparse
import ast
import os
from typing import List, Optional, Union
import re

import yaml

from chatlearn.utils.constant import RAY_PG_STRATEGY
from chatlearn.utils.logger import logger
from chatlearn.utils.utils import get_attributes


def get_path(fn, folder):
    if not fn.startswith("/") and not fn.startswith(folder):
        fn = os.path.join(folder, fn)
    assert os.path.exists(fn), f'{fn} not exists'
    return fn


def convert_type(data):
    try:
        if data == 'null':
            return None
        return ast.literal_eval(data)
    except Exception:
        return data


def parse_value(value):
    if isinstance(value, dict):
        return {k: parse_value(v) for k, v in value.items()}

    if isinstance(value, str):
        if value.strip().startswith("${"):
            # ${env_name:default_value}
            placeholder = value.replace("${", "")[:-1]
            placeholder = placeholder.split(":")
            env_name = placeholder[0]
            if env_name in os.environ:
                value = convert_type(os.environ[env_name])
            else:
                if len(placeholder) > 1:
                    value = convert_type(placeholder[1])
                else:
                    logger.warning(f"cannot find value for {env_name}, set to None")
                    value = None
        # handling scientific notation(e.g., "5e-6", "5E+10")
        elif re.match(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$", value):
            try:
                value = float(value)
            except Exception:
                pass
    return value


def update_dict(src, dst):
    # do not overwrite
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
        else:
            if isinstance(v, dict) and isinstance(dst[k], dict):
                update_dict(v, dst[k])


def parse_args_from_yaml(config_file, config_dir):
    with open(config_file, 'r', encoding='utf-8') as stream:
        config_vars = yaml.load(stream, Loader=yaml.SafeLoader)
        # empty yaml file
        if config_vars is None:
            return {}
        config_vars = {key: parse_value(value) for key, value in config_vars.items()}
        if 'includes' in config_vars:
            includes_vars = {}
            # iterate in reverse order, so the next include overwrite the prev
            for base in reversed(config_vars["includes"]):
                base_path = get_path(base, config_dir)
                base_config = parse_args_from_yaml(base_path, config_dir)
                update_dict(base_config, includes_vars)
            update_dict(includes_vars, config_vars)
        return config_vars


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='ChatLearn Arguments',
                                     allow_abbrev=False)

    parser.add_argument("-c", "--config",
                        required=False,
                        help="where to load YAML configuration",
                        metavar="FILE")

    args, _ = parser.parse_known_args()

    if args.config:
        config_dir = os.path.dirname(args.config)
        args_yaml = parse_args_from_yaml(args.config, config_dir)
    else:
        config_dir = None
        args_yaml = None
    config = Config(args_yaml, config_dir)

    return config


class BaseConfig:
    """Base class includes some common format functions."""

    def __init__(self):
        self._finalize = True

    def __str__(self):
        members = [attr for attr in dir(self) \
                   if not callable(getattr(self, attr)) and not attr.startswith("__")]
        ser_str = self.__class__.__name__ + " {\n"
        for key in members:
            if key.startswith('_'):
                continue
            attr = getattr(self, key)
            attr = '"{}"'.format(attr) if isinstance(attr, str) else attr
            ser_str += "    %s = %s,\n" % (key, attr)
        ser_str += "}"

        return ser_str

    def __repr__(self):
        return self.__str__()

    def validate(self):
        pass


class SubConfig(BaseConfig):
    """Sub Config"""
    _is_changed = False

    def __setattr__(self, name, value):
        if not name.startswith("_") and getattr(self, name) != value:
            self._is_changed = True
        super().__setattr__(name, value)

    def is_changed(self):
        return self._is_changed

class ModelConfig(BaseConfig):
    """Config for model."""
    #: [required] number of GPU used for one model, default 0
    num_gpu: int = 0
    #: [required] number of GPU used for one model, default 0
    num_cpu: int = 0
    #: [optional] gpu per process, e.g., for PyTorch DDP, Megatron, `gpu_per_process` is set to 1
    gpu_per_process: int = None
    #: [optional] cpu per process
    cpu_per_process: int = None
    #: [optional] number of module replica,
    #: for gpu model, num_replica = num_gpu // (TP * PP * DP * EP),
    #: for cpu model, num_replica = num_cpu // cpu_per_process
    # num_replica: int = 1
    #: [required] whether model is trainable
    trainable: bool = False
    #: [optional] zero size
    zero_size: int = None
    #: [optional] FSDP parallel size
    fsdp_size: int = None
    #: [optional] Sequence parallel size
    sp_size: int = None
    #: [optional] config file for model
    model_config_file: str = ""
    config_dir: str = ""
    #: [optional] placeholder for other args
    args_dict: dict = None
    #: [optional] generation batch size, will overwrite generation batch size in RuntimeConfig
    generation_batch_size: int = 1
    #: offload optimizer states
    offload_optimizer_states = False
    #: parameter sync frequency
    sync_frequency = 1
    #: offload weights
    offload_weights = False
    #: free grad buffers
    free_grad_buffers = False
    #: overall switch for offload optimizer states/weights and free grad buffers
    free_memory = False

    def __init__(self):
        super().__init__()
        self.args_dict = {}

    def __str__(self):
        members = [attr for attr in dir(self) \
                   if not callable(getattr(self, attr)) and not attr.startswith("__")]
        ser_str = self.__class__.__name__ + " {\n"
        for key in members:
            if key.startswith('_'):
                continue
            attr = getattr(self, key)
            attr = '"{}"'.format(attr) if isinstance(attr, str) else attr
            ser_str += "    %s = %s,\n" % (key, attr)
        ser_str += "}"

        return ser_str


class RuntimeConfig(BaseConfig):
    """training related configs."""

    #: [required] number of episodes. One episode includes a inference and training loop.
    num_episode: int = 5000
    #: [required] number of samples per episode.
    sample_per_episode: int = 1000
    #: [optional] number of training epoch per episode. default set to 1.
    num_training_epoch: int = 1
    #: [required] training micro batch size.
    train_micro_batch_size: int = 2
    #: [required] training global batch size.
    train_global_batch_size: int = None
    #: [required] save checkpoint per `save_episode_interval` episodes.
    save_episode_interval: int = None
    #: [required]: data_path for dataset or a List of data_path for different kind of datasets
    data_path: Optional[Union[List[str], str]] = None
    #: [optional]: the ratio for each kind of data_path in a training episode, default: None
    data_ratio: Optional[Union[List[int], int]] = None
    #: [optional]: shuffle in each epoch of dataset, default: True
    data_shuffle: Optional[bool] = True
    #: [optional]: rerank batch of data by row, default: True
    data_rerank: Optional[bool] = True
    #: [optional]: colocate models into the same device
    colocation: List[str] = []
    #: [optional]: eval every N episode, if 0, will not eval
    eval_episode_interval: int = 0
    #: [optional]: enable resume training when data checkpoint is set
    enable_resume_training: bool = True
    #: [optional]: checkpoint for dataloader
    data_checkpoint_path: str = None
    #: [optional]: max data checkpoint nums
    max_data_ckpt_nums: int = None
    #: [optional]: load data checkpoint from iteration
    load_data_checkpoint_iteration: int = None
    #: [optional]: stream_data_loader type, ["fixed", "dynamic"]
    stream_data_loader_type: str = "fixed"
    #: private
    debug: bool = False
    #: enable nsys nvtx
    nsys: bool = False
    #: profiler dir
    profiler_dir: str = None
    #: max number of replay episodes, if `max_replay_episode` is set to -1, then replay all episodes
    #: if `max_replay_episode` is set to 0, then replay is disabled
    max_replay_episode: int = 0
    #: replay after n episodes
    replay_episode_offset: int = 0
    #: consumed samples
    consumed_samples: int = 0
    #: concurrent model setup
    concurrent_setup: bool = False
    #: bucket size in the memory manager to reduce peak memory
    bucket_size_mb_in_memory_manager: int = 1024
    #: [optional] cpu only model schedule policy, PACK or SPREAD
    #: PACK: All provided bundles are packed onto a single node on a best-effort basis.
    #: SPREAD: Each bundle is spread onto separate nodes on a best-effort basis.
    cpu_schedule_strategy: str = RAY_PG_STRATEGY.SPREAD.value
    #: exp name for each run
    exp_name: str = "CHATLEARN"
    #: output dir
    output_dir: str = "./"
    #: whether to eval before training
    enable_eval_before_training: bool = False
    #: policy to regroup queue
    policy_to_regroup_queue: str = "global_barrier"
    #: configuration file path for logging
    log_config_file: str = ""
    #: [optional] placeholder for log_args_dict
    log_args_dict: dict = None

    def __init__(self):
        super().__init__()
        self._args_dict = {}

    def get(self, key):
        """
        Get other config by key.

        Args
        ----
        key: str
            key to get config
        """
        if key not in self._args_dict:
            logger.warning(f"{key} not found in RuntimeConfig")
        else:
            return self._args_dict[key]

    def validate(self):
        """
        :meta private:
        """

class RuntimeEnvConfig(BaseConfig):
    """Runtime env config, you can refer https://docs.ray.io/en/latest/ray-core/handling-dependencies.html for more information."""

    #: pip install packages
    pip: List[str] = []
    #: python modules
    py_modules: List[str] = []
    #: working directory
    # working_dir: str = os.getcwd()
    working_dir: str = None
    #: platform, e.g., DLC
    platform: str = ""
    #: excludes files from packaging
    excludes: List[str] = []

    def __init__(self):
        super().__init__()
        self._args_dict = {}

    def get(self, key):
        """
        Get other config by key

        Args
        ----
        key: str
            Key to get config.
        """
        if key not in self._args_dict:
            logger.warning(f"{key} not found in RuntimeConfig")
        else:
            return self._args_dict[key]


class Config(BaseConfig):
    """A class to manage chatlearn configuration.

    Args
    ----
      param_dict: dict
      dict format of parameters."""

    def __init__(self, param_dict=None, config_dir=None):
        super().__init__()
        self._finalize = False
        self.models = {}
        self.env_args = RuntimeEnvConfig()
        self.runtime_args = RuntimeConfig()
        self.config_dir = config_dir

        if param_dict:
            self._parse_params(param_dict)
            self._validate_params()

    def _parse_params(self, param_dict):
        """Parse params from param_dict."""

        def set_param(user_args, config_cls, instance):
            for attribute, default_value in get_attributes(config_cls):
                if attribute in user_args:
                    value = user_args[attribute]
                    if attribute == "colocation":
                        colocation_list = []
                        for group in value:
                            colocation_list.append(group.replace(' ', '').split(','))
                        value = colocation_list
                    elif attribute == "data_ratio":
                        if isinstance(value, str):
                            value = [int(v) for v in value.split(',')]
                else:
                    value = default_value
                original_value = getattr(instance, attribute)
                if original_value is not None:
                    assert isinstance(original_value, type(value)), \
                        f"{instance}.{attribute} should be type of {type(original_value)} but got {type(value)}"

                setattr(instance, attribute, value)
            for user_attribute in user_args:
                if not hasattr(config_cls, user_attribute):
                    if hasattr(instance, "_args_dict"):
                        getattr(instance, "_args_dict")[user_attribute] = user_args[user_attribute]
                    else:
                        raise RuntimeError(f"attribute {user_attribute} not defined in {config_cls.__name__}")
            instance.validate()

        for model_name, model_args in param_dict["models"].items():
            model_config = ModelConfig()
            model_config.config_dir = self.config_dir
            for user_attribute, user_value in model_args.items():
                if hasattr(ModelConfig, user_attribute):
                    original_value = getattr(ModelConfig, user_attribute)
                    if original_value is not None:
                        assert isinstance(user_value, type(original_value)), \
                            f"ModelConfig.{user_attribute} should be type of {type(original_value)} but got {type(user_value)} ({user_value})"
                    setattr(model_config, user_attribute, user_value)
                else:
                    logger.warning(f"unknown argument {user_attribute}")

            self.models[model_name] = model_config
            if model_config.model_config_file:
                model_config.model_config_file = get_path(model_config.model_config_file, self.config_dir)
                model_config.args_dict = parse_args_from_yaml(model_config.model_config_file, self.config_dir)
        if "runtime" in param_dict:
            set_param(param_dict["runtime"], RuntimeConfig, self.runtime_args)
        elif "rlhf" in param_dict:
            logger.warning("rlhf is deprecated, please use runtime as section name")
            set_param(param_dict["rlhf"], RuntimeConfig, self.runtime_args)
        if "runtime_env" in param_dict:
            set_param(param_dict["runtime_env"], RuntimeEnvConfig, self.env_args)

        if self.runtime_args.log_config_file:
            self.runtime_args.log_config_file = get_path(self.runtime_args.log_config_file, self.config_dir)
            self.runtime_args.log_args_dict = parse_args_from_yaml(self.runtime_args.log_config_file, self.config_dir)

        def _get_and_check_type(value, default_value, key):
            # To be noticed: all str type values should in lower case.
            if isinstance(value, str):
                value = value.lower()
            if default_value is None:
                return value
            if not isinstance(value, type(default_value)):
                raise ValueError("%s type error, expected: %s." \
                                 % (key, type(default_value)))
            return value

    def _validate_params(self):
        if self.runtime_args.train_global_batch_size is None:
            self.runtime_args.train_global_batch_size = self.runtime_args.train_micro_batch_size
        assert self.runtime_args.train_global_batch_size % self.runtime_args.train_micro_batch_size == 0, \
            f"train_global_batch_size should be times of train_micro_batch_size," \
            f"but got {self.runtime_args.train_global_batch_size}/{self.runtime_args.train_micro_batch_size}"
        assert self.runtime_args.train_global_batch_size <= self.runtime_args.sample_per_episode, \
            "train_global_batch_size should be less than or equal to sample_per_episode, " \
            f"got {self.runtime_args.train_global_batch_size} and {self.runtime_args.sample_per_episode}"
        assert self.runtime_args.stream_data_loader_type.lower() in ["fixed", "dynamic"]
        assert self.runtime_args.cpu_schedule_strategy in [strategy.value for strategy in RAY_PG_STRATEGY]
        if isinstance(self.runtime_args.data_path, list):
            assert self.runtime_args.data_ratio is not None and isinstance(self.runtime_args.data_ratio, list), (
                f"expect data_ratio to be list when data_path is list, got {self.runtime_args.data_ratio}"
            )
            assert len(self.runtime_args.data_path) == len(self.runtime_args.data_ratio), (
                "expect data_path and data_ratio to have same length, "
                f"got {len(self.runtime_args.data_path)} and {len(self.runtime_args.data_ratio)}"
            )

        # TODO: check the following assertions
        for model_name, model_args in self.models.items():
            if model_args.num_gpu >= 1:
                if model_args.gpu_per_process is None:
                    model_args.gpu_per_process = 1
                else:
                    assert model_args.gpu_per_process <= model_args.num_gpu, \
                        f"{model_name}: gpu_per_process: {model_args.gpu_per_process}, num_cpu: {model_args.num_gpu}"
            elif model_args.num_cpu >= 1:
                if model_args.cpu_per_process is None:
                    model_args.cpu_per_process = 1
                else:
                    assert model_args.cpu_per_process <= model_args.num_cpu, \
                        f"{model_name}: cpu_per_process: {model_args.cpu_per_process}, num_cpu: {model_args.num_cpu}"

            for key in ["pipeline_model_parallel_size", "tensor_model_parallel_size", "zero_size", "sp_size"]:
                if model_args.args_dict.get(key) is not None:
                    setattr(model_args, key, model_args.args_dict.get(key))
                    assert getattr(model_args, key) >= 1
                elif getattr(model_args, key) is None:
                    setattr(model_args, key, 1)

            for key in ["fsdp_size"]:
                if getattr(model_args, key) is not None:
                    setattr(model_args, key, getattr(model_args, key))
                    if getattr(model_args, key) == -1:
                        print(f"set_fsdp_size {getattr(model_args, key)} to num_gpu: {model_args.num_gpu}")
                        setattr(model_args, key, model_args.num_gpu)
                    assert getattr(model_args, key) >= 1
                elif getattr(model_args, key) is None:
                    setattr(model_args, key, 1)

            ep_size = model_args.args_dict.get("expert_model_parallel_size")
            moe_ep_size = model_args.args_dict.get("moe_expert_model_parallel_size")
            if ep_size is not None and moe_ep_size is not None:
                assert ep_size == moe_ep_size, (
                    f"{model_name}: if you set moe_expert_model_parallel_size ({moe_ep_size}), "
                    f"it must be equal to expert_model_parallel_size ({ep_size})"
                )
                finalized_ep_size = ep_size
            elif ep_size is not None:
                finalized_ep_size = ep_size
            elif moe_ep_size is not None:
                finalized_ep_size = moe_ep_size
            else:
                finalized_ep_size = 1
            assert finalized_ep_size >= 1
            setattr(model_args, "expert_model_parallel_size", finalized_ep_size)

            if model_args.tensor_model_parallel_size > 1 or model_args.pipeline_model_parallel_size > 1 or model_args.expert_model_parallel_size > 1:
                assert model_args.zero_size == 1 or model_args.zero_size is None
                assert model_args.fsdp_size == 1 or model_args.fsdp_size is None
                assert model_args.num_gpu % (
                    model_args.tensor_model_parallel_size * model_args.pipeline_model_parallel_size * model_args.expert_model_parallel_size) == 0, \
                    f"{model_name}: num_gpu must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size * " \
                    f"expert_model_parallel_size, but got num_gpu = {model_args.num_gpu}, " \
                    f"tensor_model_parallel_size = {model_args.tensor_model_parallel_size}, " \
                    f"pipeline_model_parallel_size = {model_args.pipeline_model_parallel_size}, and "\
                    f"expert_model_parallel_size = {model_args.expert_model_parallel_size}."
            assert model_args.num_gpu > 0 or model_args.num_cpu > 0, \
                f"{model_name} num_gpu: {model_args.num_gpu}, num_cpu: {model_args.num_cpu}, at least one of them should be set"

            if model_args.free_memory:
                model_args.offload_weights = True
                if model_args.trainable:
                    model_args.free_grad_buffers = True
                    model_args.offload_optimizer_states = True
        if self.runtime_args.colocation and len(self.runtime_args.colocation) > 0:
            model_set = set()
            for colocate_models in self.runtime_args.colocation:
                for model_name in colocate_models:
                    assert model_name not in model_set, f"Model {model_name} should only appear once in colocation group"
                    model_set.add(model_name)
        if self.runtime_args.exp_name not in self.runtime_args.output_dir:
            self.runtime_args.output_dir = f"{self.runtime_args.output_dir}/{self.runtime_args.exp_name}"
        logger.info(f"Env Config: \n{self.env_args}")
        logger.info(f"Runtime Config: \n{self.runtime_args}")
        for name, model_args in self.models.items():
            logger.info(f"Model({name}) Config: \n{model_args}")
