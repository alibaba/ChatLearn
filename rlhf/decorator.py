import torch
import time
import ray
import traceback
from rlhf.logger import logger
from rlhf import utils



def monitor_error(func, func_name):

    def inner(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"catch exception ========= in {self.name} {e}, {traceback.format_exc()}")
            ray.get(self.error_signal.set.remote())
            raise
    return inner


def timeit(func, func_name):

    def inner(self, *args, **kwargs):
        if self.rank == 0:
            # for the class inherited from base, it may call multiple times, so use the first start time
            if not self.timers(func_name).started_:
                self.timers(func_name).start()
            ret = func(self, *args, **kwargs)
            self.timers(func_name).stop()
        else:
            ret = func(self, *args, **kwargs)
        return ret

    return inner



def preprocess_compute(func, merge_input):
    """
    1. if merge_input is True, merge a list of dict into one dict, i.e., merge inputs of forward_step.
    2. split a list of data for data_parallel, this is used for train_step
    3. convert output to cpu
    """
    def inner(self, *args, **kwargs):
        args = utils.get(args)
        if merge_input and len(args) > 1:
            if all(isinstance(arg, dict) for arg in args):
                merged = {}
                for arg in args:
                    merged.update(arg)
                args = [merged]
        if self.data_parallel_size is not None and \
                self.data_parallel_rank is not None and \
                self.data_parallel_size > 1:
            data_list = args[0]
            assert isinstance(data_list, list)
            start_idx, end_idx = utils.split_index(len(data_list), self.data_parallel_size)[self.data_parallel_rank]
            args = list(args)
            sub_data_list = data_list[start_idx: end_idx]
            args[0] = sub_data_list
        ret = func(self, *args, **kwargs)
        ret = utils.to_device('cpu', ret)
        return ret

    return inner


def decorate_class_func(cls, func_name, decorator, *args, **kwargs):
    func = getattr(cls, func_name)
    if func.__qualname__.startswith(decorator.__name__):
        # already decorated
        logger.warn(f"{func_name} {func} already decorated with {decorator}")
        return
    setattr(cls, func_name, decorator(func, *args, **kwargs))
