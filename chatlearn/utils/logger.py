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
"""logging"""

import logging
import os

import torch
from tqdm import tqdm


def setup_logger(log_level=None, model_name=None, ip_addr=None):
    logger_name = "ChatLearn" if model_name is None else f"ChatLearn-{model_name}"
    _logger = logging.getLogger(logger_name)
    _logger.handlers.clear()
    _logger.propagate = False
    if log_level is None:
        log_level = logging.INFO
    _logger.setLevel(log_level)
    handler = logging.StreamHandler()
    if ip_addr is None:
        handler.setFormatter(logging.Formatter(
            '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'))
    else:
        rank = os.environ.get("RANK", 0)
        handler.setFormatter(logging.Formatter(
            f"[%(asctime)s %(name)s {ip_addr} RANK:{rank}] (%(filename)s %(lineno)d): %(levelname)s %(message)s"))
    handler.setLevel(log_level)
    _logger.addHandler(handler)
    return _logger


logger = setup_logger()


def log_rank_0(msg, custom_logger=None):
    _logger = custom_logger if custom_logger is not None else logger
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            _logger.info(msg)
    else:
        _logger.info(msg)


def debug_rank_0(msg, custom_logger=None):
    _logger = custom_logger if custom_logger is not None else logger
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            _logger.debug(msg)
    else:
        _logger.debug(msg)


class logging_tqdm(tqdm):
    """logging tqdm"""

    def __init__(
        self,
        *args,
        tqdm_logger=None,
        mininterval: float = 1,
        bar_format: str = '{desc}{percentage:3.0f}%{r_bar}',
        desc: str = 'progress: ',
        **kwargs):
        self._logger = tqdm_logger
        super().__init__(
            *args,
            mininterval=mininterval,
            bar_format=bar_format,
            desc=desc,
            **kwargs
        )

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        return logger

    def display(self, msg=None, pos=None): # pylint: disable=unused-argument
        if not self.n:
            # skip progress bar before having processed anything
            return
        if not msg:
            msg = f"{self}"
        self.logger.info('%s', msg)
