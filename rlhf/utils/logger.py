import logging

import torch
from tqdm import tqdm

logger = logging.getLogger("RLHF")

def setup_logger(log_level=None):
    if log_level is None:
        log_level = logging.INFO
    global logger
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
            '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'))
    handler.setLevel(log_level)
    logger.addHandler(handler)
    return logger

setup_logger()

def log_rank_0(msg):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(msg)
    else:
        logger.info(msg)

class logging_tqdm(tqdm):

    def __init__(
            self,
            *args,
            logger=None,
            mininterval: float = 1,
            bar_format: str = '{desc}{percentage:3.0f}%{r_bar}',
            desc: str = 'progress: ',
            **kwargs):
        self._logger = logger
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


    def display(self, msg=None, pos=None):
        if not self.n:
            # skip progress bar before having processed anything
            return
        if not msg:
            msg = self.__str__()
        self.logger.info('%s', msg)
