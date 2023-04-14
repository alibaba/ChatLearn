import logging
import torch

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
