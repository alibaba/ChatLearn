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
"""arguments from command or yaml."""

import argparse
import ast
import os

import yaml

from rlhf.utils.logger import logger
from rlhf.utils.utils import get_attributes


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
        config_vars = yaml.load(stream, Loader=yaml.FullLoader)
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
    parser = argparse.ArgumentParser(description='RLHF Arguments',
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


class ModelConfig(BaseConfig):
    """Config for model."""

    #: [required] number of device used for one model
    num_device = 1
    #: [required] gpu per process, e.g., for PyTorch DDP, Megatron, DeepSpeed, `gpu_per_process` is set to 1
    gpu_per_process = 1
    #: [required] whether model is trainable
    trainable = False
    #: [optional] tensor model parallel size
    tensor_model_parallel_size = None
    #: [optional] pipeline model parallel size
    pipeline_model_parallel_size = None
    #: [optional] config file for model
    model_config_file = ""
    config_dir = ""
    #: [optional] model type, e.g., Torch/Tensorflow, etc
    model_type = ""
    #: [optional] placeholder for other args
    args_dict = {}
    #: [optional] generation batch size, will overwrite generation batch size in RLHFConfig
    generation_batch_size = -1
    #: [optional] return rlhf data
    return_rlhf_data = False


class RLHFConfig(BaseConfig):
    """RLHF config"""

    #: [required] number of ppo episodes. One episode includes a inference and training loop.
    num_ppo_episode = 5000
    #: [required] number of samples per episode.
    sample_per_episode = 1000
    #: [optional] number of training epoch per episode. default set to 1.
    num_training_epoch = 1
    #: [required] generation(inference) batch size.
    generation_batch_size = 2
    #: [required] training micro batch size.
    train_micro_batch_size = 2
    #: [required] training global batch size.
    train_global_batch_size = None
    #: [required] save checkpoint per `save_episode_interval` episodes.
    save_episode_interval = None
    #: [optional] log time and memory per `log_interval` iterations.
    log_interval = 1
    #: [required]: data_path for dataset
    data_path = None
    #: [optional]: colocate models into the same device
    colocation = []
    #: [optional]: eval every N episode, if 0, will not eval
    eval_episode_interval = 0
    #: [optional]: checkpoint for dataloader
    data_checkpoint_path = None
    #: [optional]: max data checkpoint nums
    max_data_ckpt_nums = None
    #: [optional]: load data checkpoint from iteration
    load_data_checkpoint_iteration = None
    #: [optional]: dynamic training samples
    dynamic_train_samples = False
    #: private
    debug = False
    #: enable nsys nvtx
    nsys = False
    #: coalesce parameters in model sync
    coalesce_param = True
    #: coalesce_buffer size in mb
    coalesced_buffer_mb = 100

    def __init__(self):
        super().__init__()
        self._args_dict = {}

    def get(self, key):
        """
        get other config by key

        Args:
            key: key to get config
        """
        if key not in self._args_dict:
            logger.warning(f"{key} not found in RLHFConfig")
        else:
            return self._args_dict[key]

    def validate(self):
        for key in self._args_dict:
            if key == "save_interval":
                raise Exception("save_interval is deprecated, please use save_episode_interval to save checkpoints")


class RuntimeEnvConfig(BaseConfig):
    """runtime env config, you can refer https://docs.ray.io/en/latest/ray-core/handling-dependencies.html for more information."""

    #: pip install packages
    pip = []
    #: python modules
    py_modules = []
    #: working directory
    working_dir = os.getcwd()
    #: platform, e.g., DLC
    platform = ""
    #: excludes files from packaging
    excludes = []

    def __init__(self):
        super().__init__()
        self._args_dict = {}

    def get(self, key):
        """
        get other config by key

        Args:
            key: key to get config
        """
        if key not in self._args_dict:
            logger.warning(f"{key} not found in RuntimeConfig")
        else:
            return self._args_dict[key]


class Config(BaseConfig):
    """A class to manage epl configuration.

    Args:
      param_dict: Dict format of parameters."""

    def __init__(self, param_dict=None, config_dir=None):
        super().__init__()
        self._finalize = False
        self.models = {}
        self.env_args = RuntimeEnvConfig()
        self.rlhf_args = RLHFConfig()
        self.config_dir = config_dir

        self.initialized = False
        if param_dict:
            self._parse_params(param_dict)
            self._validate_params()
        self._finalize = True

    def _parse_params(self, param_dict):
        """Parse params from param_dict."""

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

        def set_param(namespace, config_cls, instance):
            if namespace not in param_dict:
                return
            user_args = param_dict[namespace]
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
                    getattr(instance, "_args_dict")[user_attribute] = user_args[user_attribute]
            instance.validate()

        set_param("rlhf", RLHFConfig, self.rlhf_args)
        set_param("runtime_env", RuntimeEnvConfig, self.env_args)

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
        if self.rlhf_args.train_global_batch_size is None:
            self.rlhf_args.train_global_batch_size = self.rlhf_args.train_micro_batch_size
        assert self.rlhf_args.train_global_batch_size % self.rlhf_args.train_micro_batch_size == 0, \
            f"train_global_batch_size should be times of train_micro_batch_size," \
            f"but got {self.rlhf_args.train_global_batch_size}/{self.rlhf_args.train_micro_batch_size}"

        for _, model_args in self.models.items():
            assert model_args.gpu_per_process <= model_args.num_device
            if model_args.generation_batch_size is None or model_args.generation_batch_size <= 0:
                if self.rlhf_args.generation_batch_size:
                    model_args.generation_batch_size = self.rlhf_args.generation_batch_size
            for key in ["pipeline_model_parallel_size", "tensor_model_parallel_size"]:
                if model_args.args_dict.get(key) is not None:
                    setattr(model_args, key, model_args.args_dict.get(key))
                    assert getattr(model_args, key) >= 1
                else:
                    setattr(model_args, key, 1)
            assert model_args.num_device % (
                model_args.tensor_model_parallel_size * model_args.pipeline_model_parallel_size) == 0
            model_args.num_replica = model_args.num_device // (
                model_args.tensor_model_parallel_size * model_args.pipeline_model_parallel_size)
