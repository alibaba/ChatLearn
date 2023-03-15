import logging

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
