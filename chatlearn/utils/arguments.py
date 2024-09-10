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
from typing import List

import yaml

from chatlearn.utils.constant import LORA_LAYER, RAY_PG_STRATEGY, PARAM_SYNC_COMM_TYPE
from chatlearn.utils.logger import logger
from chatlearn.utils.utils import get_attributes


def get_path(fn, folder):
    if not fn.startswith("/") and not fn.startswith(folder):
        fn = os.path.join(folder, fn)
    assert os.path.exists(fn), f'{fn} not exists'
    return fn


def convert_type(data):
    try:
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


class LoraConfig(SubConfig):
    """Config for lora"""
    #: enable lora, default False.
    enable_lora: bool = False
    #: The "name_scope" parameter is used to specify a particular module to be converted to its LoRA.
    #: By default, it is set to None, which means there is no restriction on the module and any module
    #: can be converted using the "lora_layer" parameter. However, if "name_scope" is set to a specific
    #: value (e.g., "encoder"), only the modules whose name_scope contains the value "encoder" will be converted to LoRA.
    part_module_name: str = None
    #: The rank value of the LoRA, which is the r dimension of the A/B matrix.
    lora_dim: int = 8
    #: The LoRA dropout ratio refers to whether dropout computation is inserted in the forward pass
    #: of the LoRA layer. By default, the dropout ratio is set to 0.0.
    lora_dropout: float = 0.0
    #: When adding the values of the LoRA A and B matrices to the original weight matrix,
    #: the scaling value is set as "W = W + A * B * lora_scaling". By default, the scaling value
    #: is set to 1.0.
    lora_scaling: float = 1.0
    #: The layer class names involved in LoRA training in the model, separated by commas.
    lora_layer: str = LORA_LAYER
    #: LoRA training is enabled only in the ColumnParallelLinear layer of the MHA QKV module.
    column_only_qkv: bool = False


class BatchGenerationConfig(SubConfig):
    """Config for batch generation ranking and memory-efficiency."""

    #: [optional] sort prompts by length each episode.
    ranking: bool = False
    #: [optional] min prompt length in the first stage of batch generation.
    min_prompt_length: int = 0


class ModelConfig(BaseConfig):
    """Config for model."""

    #: [legacy] number of GPU used for one model, default 0.
    num_device: int = 0
    #: [required] number of GPU used for one model, default 0, same as num_device
    num_gpu: int = 0
    #: [required] number of GPU used for one model, default 0
    num_cpu: int = 0
    #: [optional] gpu per process, e.g., for PyTorch DDP, Megatron, DeepSpeed, `gpu_per_process` is set to 1
    gpu_per_process: int = None
    #: [optional] cpu per process
    cpu_per_process: int = None
    #: [optional] number of module replica,
    #: for gpu model, num_replica = num_gpu // (TP * PP * DP),
    #: for cpu model, num_replica = num_cpu // cpu_per_process
    num_replica: int = 1
    #: [required] whether model is trainable
    trainable: bool = False
    #: [optional] tensor model parallel size
    tensor_model_parallel_size: int = None
    #: [optional] pipeline model parallel size
    pipeline_model_parallel_size: int = None
    #: [optional] zero size
    zero_size: int = None
    #: [optional] config file for model
    model_config_file: str = ""
    config_dir: str = ""
    #: [optional] model type, e.g., Torch/Tensorflow, etc
    model_type: str = ""
    #: [optional] placeholder for other args
    args_dict: dict = None
    #: [optional] generation batch size, will overwrite generation batch size in RuntimeConfig
    generation_batch_size: int = -1
    #: lora config
    lora: LoraConfig = None
    #: batch generation config
    batch_generation: BatchGenerationConfig = None
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
        self.lora = LoraConfig()
        self.batch_generation = BatchGenerationConfig()

    def __str__(self):
        members = [attr for attr in dir(self) \
                   if not callable(getattr(self, attr)) and not attr.startswith("__")]
        ser_str = self.__class__.__name__ + " {\n"
        for key in members:
            if key.startswith('_'):
                continue
            attr = getattr(self, key)
            if key in ["lora", "batch_generation"]:
                if not attr.is_changed():
                    continue
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
    #: [required] generation(inference) batch size.
    generation_batch_size: int = 2
    #: [required] training micro batch size.
    train_micro_batch_size: int = 2
    #: [required] training global batch size.
    train_global_batch_size: int = None
    #: [required] save checkpoint per `save_episode_interval` episodes.
    save_episode_interval: int = None
    #: [optional] log time and memory per `log_interval` iterations.
    log_interval: int = 1
    #: [required]: data_path for dataset
    data_path: str = None
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
    #: coalesce parameters in model sync
    coalesce_param: bool = True
    #: coalesce_buffer size in mb
    coalesced_buffer_mb: int = 100
    #: concurrent parameter sync
    concurrent_comm: bool = True
    #: parameter sync communication type, broadcast/p2p
    param_sync_comm_type: str = PARAM_SYNC_COMM_TYPE.BROADCAST.value
    #: parameter sync max workers
    param_sync_max_workers: int = None
    #: max number of relay episodes, if `max_relay_episode` is set to -1, then relay all episodes
    #: if `max_relay_episode` is set to 0, then relay is disabled
    max_relay_episode: int = 0
    #: relay after n episodes
    relay_episode_offset: int = 0
    #: consumed samples
    consumed_samples: int = 0
    #: concurrent model setup
    concurrent_setup: bool = False
    #: bucket size in the memory manager to reduce peak memory
    bucket_size_mb_in_memory_manager: int = 1024
    #: free collective group after parameter synchronization and rebuild before next synchronization
    free_sync_collective_group: bool = False
    #: [optional] cpu only model schedule policy, PACK or SPREAD
    #: PACK: All provided bundles are packed onto a single node on a best-effort basis.
    #: SPREAD: Each bundle is spread onto separate nodes on a best-effort basis.
    cpu_schedule_strategy: str = RAY_PG_STRATEGY.SPREAD.value
    #: exp name for each run
    exp_name: str = "CHATLEARN"
    #: output dir
    output_dir: str = "./"

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
        for key in self._args_dict:
            if key == "save_interval":
                raise Exception("save_interval is deprecated, please use save_episode_interval to save checkpoints")


class RuntimeEnvConfig(BaseConfig):
    """Runtime env config, you can refer https://docs.ray.io/en/latest/ray-core/handling-dependencies.html for more information."""

    #: pip install packages
    pip: List[str] = []
    #: python modules
    py_modules: List[str] = []
    #: working directory
    working_dir: str = os.getcwd()
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
        self._active_module_args = None

        self.initialized = False
        if param_dict:
            self._parse_params(param_dict)
            self._validate_params()
        # remove later, just for compatibility
        self.rlhf_args = self.runtime_args
        self._finalize = True

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
                    if 'num_device' == user_attribute:
                        logger.warning("num_device is deprecated, please use num_gpu instead")
                        if 'num_gpu' not in model_args.keys():
                            setattr(model_config, "num_gpu", user_value)
                        else:
                            logger.warning("both num_device and num_gpu are set, use num_gpu")
                            continue
                    if 'lora' == user_attribute:
                        set_param(user_value, LoraConfig, model_config.lora)
                        user_value = model_config.lora
                    elif "batch_generation" == user_attribute:
                        set_param(user_value, BatchGenerationConfig, model_config.batch_generation)
                        user_value = model_config.batch_generation
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
        assert self.runtime_args.stream_data_loader_type.lower() in ["fixed", "dynamic"]
        assert self.runtime_args.cpu_schedule_strategy in [strategy.value for strategy in RAY_PG_STRATEGY]
        assert self.runtime_args.param_sync_comm_type in list(PARAM_SYNC_COMM_TYPE)
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
            if model_args.generation_batch_size is None or model_args.generation_batch_size <= 0:
                if self.runtime_args.generation_batch_size:
                    model_args.generation_batch_size = self.runtime_args.generation_batch_size
            for key in ["pipeline_model_parallel_size", "tensor_model_parallel_size", "zero_size"]:
                if model_args.args_dict.get(key) is not None:
                    setattr(model_args, key, model_args.args_dict.get(key))
                    assert getattr(model_args, key) >= 1
                elif getattr(model_args, key) is None:
                    setattr(model_args, key, 1)
            if model_args.tensor_model_parallel_size > 1 or model_args.pipeline_model_parallel_size > 1:
                assert model_args.zero_size == 1 or model_args.zero_size is None
                assert model_args.num_gpu % (
                    model_args.tensor_model_parallel_size * model_args.pipeline_model_parallel_size) == 0, \
                    "num_gpu must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size " \
                    f"for {model_name} model, but got num_gpu = {model_args.num_gpu}" \
                    f"tensor_model_parallel_size = {model_args.tensor_model_parallel_size}, and " \
                    f"pipeline_model_parallel_size = {model_args.pipeline_model_parallel_size}."
            assert model_args.num_gpu > 0 or model_args.num_cpu > 0, \
                f"{model_name} num_gpu: {model_args.num_gpu}, num_cpu: {model_args.num_cpu}, at least one of them should be set"

            if model_args.num_gpu >= 1:
                if model_args.zero_size > 1:
                    assert model_args.num_gpu % model_args.zero_size == 0
                    model_args.num_replica = model_args.num_gpu // model_args.zero_size
                else:
                    model_args.num_replica = model_args.num_gpu // (
                        model_args.tensor_model_parallel_size * model_args.pipeline_model_parallel_size)
            elif model_args.num_cpu >= 1:
                model_args.num_replica = model_args.num_cpu // model_args.cpu_per_process
            assert model_args.num_replica * model_args.generation_batch_size <= self.runtime_args.sample_per_episode, \
                f"num_replica * batch_size {model_args.num_replica}*{model_args.generation_batch_size} " + \
                f"should be less than sample_per_episode {self.runtime_args.sample_per_episode}"
            if model_args.batch_generation.min_prompt_length:
                logger.info(f"Enable batch generation: \
                    min_prompt_length = {model_args.batch_generation.min_prompt_length}")
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

    @property
    def active_module_args(self):
        return self._active_module_args

    @active_module_args.setter
    def active_module_args(self, config):
        self._active_module_args = config
