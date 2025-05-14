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
"""metric manager"""
import traceback

import wandb
from torch.utils.tensorboard import SummaryWriter

from chatlearn.utils.constant import LOG_START
from chatlearn.utils.logger import logger

class MetricManager:
    """Metric manager"""
    def __init__(self, global_args):
        self.global_args = global_args
        self.runtime_args = global_args.runtime_args
        self._setup_tensorboard()
        self._setup_wandb()

        self.writer_dict = {}
        if self.tensorboard_writer:
            self.writer_dict['tensorboard'] = self.tensorboard_writer
        if self.wandb_writer:
            self.writer_dict['wandb'] = self.wandb_writer

    def _setup_tensorboard(self):
        if (
            self.runtime_args.log_args_dict is None
            or 'enable_tensorboard' not in self.runtime_args.log_args_dict
            or not self.runtime_args.log_args_dict['enable_tensorboard']
        ):
            self.tensorboard_writer = None
            logger.info("tensorboard is disabled in engine.")
            return

        try:
            self.tensorboard_writer = SummaryWriter(
                log_dir=self.runtime_args.log_args_dict['tensorboard_dir'],
                max_queue=99999
            )
        except Exception:
            self.tensorboard_writer = None
            logger.warning(f"{LOG_START} setup tensorboard failed, tensorboard_writer is set to empty.")
        else:
            logger.info(f"{LOG_START} setup tensorboard success.")

    def _setup_wandb(self):
        if (
            self.runtime_args.log_args_dict is None
            or 'enable_wandb' not in self.runtime_args.log_args_dict
            or not self.runtime_args.log_args_dict['enable_wandb']
        ):
            self.wandb_writer = None
            logger.info("wandb is disabled in engine.")
            return

        try:
            wandb_kwargs = {
                'dir': self.runtime_args.log_args_dict['wandb_dir'],
                'project': self.runtime_args.log_args_dict['wandb_project'],
                'id': self.runtime_args.log_args_dict['wandb_id'],
                'name': self.runtime_args.log_args_dict['wandb_name'],
                'resume': self.runtime_args.log_args_dict['wandb_resume'],
                'config': self.global_args,
            }
            logger.info(f"WANDB_ARGS: {wandb_kwargs}")
            wandb.init(**wandb_kwargs)
        except Exception:
            traceback.print_exc()
            self.wandb_writer = None
            logger.warning(f"{LOG_START} setup wandb failed, wandb_writer is set to empty.")
        else:
            self.wandb_writer = wandb
            logger.info(f"{LOG_START} setup wandb success.")

    def log(self, prefix:str, global_step:int, scalar_dict):
        prefix = prefix.rstrip('/')
        logger.info(f"step {global_step} prefix {prefix}: logging metric {scalar_dict}")
        for writer_name, _ in self.writer_dict.items():
            if writer_name == 'tensorboard':
                self._tensorboard_scalar_dict(prefix, global_step, scalar_dict)
            if writer_name == 'wandb':
                self._wandb_scalar_dict(prefix, global_step, scalar_dict)

    def _tensorboard_scalar_dict(self, prefix, global_step, scalar_dict):
        if isinstance(scalar_dict, (float, int)):
            name = prefix
            value = scalar_dict
            self.tensorboard_writer.add_scalar(name, value, global_step)
        else:
            for key, value in scalar_dict.items():
                name = f"{prefix}/{key}".lstrip('/')
                self.tensorboard_writer.add_scalar(name, value, global_step)

    def _wandb_scalar_dict(self, prefix, global_step, scalar_dict):
        if isinstance(scalar_dict, (float, int)):
            name = prefix
            value = scalar_dict
            self.wandb_writer.log({f"{name}": value}, step=global_step)
        else:
            scalar_dict_with_prefix = {}
            for key, value in scalar_dict.items():
                name = f"{prefix}/{key}".lstrip('/')
                scalar_dict_with_prefix[name] = value
            self.wandb_writer.log(scalar_dict_with_prefix, step=global_step)

    def stop(self):
        if self.wandb_writer:
            self.wandb_writer.finish()
