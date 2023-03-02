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
import yaml
import inspect
import logging
from rlhf.utils import get_attributes

def parse_args_from_yaml(config_file):
    with open(config_file, 'r', encoding='utf-8') as stream:
        config_vars = yaml.load(stream, Loader=yaml.FullLoader)
        return config_vars


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='RLHF Arguments',
                                     allow_abbrev=False)

    parser.add_argument("-c", "--config",
                        required=True,
                        help="where to load YAML configuration",
                        metavar="FILE")

    args = parser.parse_args()

    if args.config:
        args_yaml = parse_args_from_yaml(args.config)
    config = Config(args_yaml)
    return config


class BaseConfig(object):
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

  def __setattr__(self, name, value):
    """Avoid adding new attributes by users."""
    if name != "_finalize" and self._finalize and not hasattr(self, name):
      raise AttributeError('{} instance has no attribute {!r}'.format(type(self).__name__, name))
    super(BaseConfig, self).__setattr__(name, value)


class ModelConfig(BaseConfig):
    """Config for model."""
  
    num_device = 1
    gpu_per_process = 1
    trainable = False
    model_type = ""
    model_config_file = ""


class RLHFConfig:
    num_rollout_worker = 1
    num_ppo_iteration = 5000
    sample_per_episode = 1000
    num_training_epoch = 1
    generation_batch_size = 2
    train_batch_size = 2


class RuntimeEnvConfig:
    pip = []
    py_modules = []
    working_dir = "./"
    # platform, e.g., DLC
    platform = ""


def _get_attributes(cls):
  """Get attributes from class."""
  return [(name, attr) for name, attr in inspect.getmembers(cls) if not name.startswith('_')]


class Config(BaseConfig):
  """A class to manage epl configuration.

  Args:
    param_dict: Dict format of parameters."""

  def __init__(self, param_dict=None):
    super().__init__()
    self._finalize = False
    self.models = {}
    self.env_args = RuntimeEnvConfig()
    self.rlhf_args = RLHFConfig()

    self._parse_params(param_dict)
    self._finalize = True
    self._validate_params()

  def _parse_params(self, param_dict):
    """Parse params from param_dict."""

    for model_name, model_args in param_dict["models"].items():
        model_config = ModelConfig()
        for user_attribute, user_value in model_args.items():
            if hasattr(ModelConfig, user_attribute):
                setattr(model_config, user_attribute, user_value)
            else:
                logging.warn(f"unknown argument {user_attribute}")

        self.models[model_name] = model_config


    def set_param(namespace, config_cls, instance):
        if namespace not in param_dict:
            return
        user_args = param_dict[namespace]
        for attribute, default_value in _get_attributes(config_cls):
            if attribute in user_args:
                value = user_args[attribute]
            else:
                value = default_value
            setattr(instance, attribute, value)
        for user_attribute in user_args:
            if not hasattr(config_cls, user_attribute):
                logging.warn(f"unknown argument {user_attribute}")


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
      pass
