import ray
from rlhf.utils.logger import logging_tqdm
from rlhf.utils.utils import flatten


def wait(refs, desc=None):
    """
    wait until all computation finish
    """
    if len(refs) == 0:
        return
    refs = flatten(refs)
    if desc is not None:
        pbar = logging_tqdm(total=len(refs), desc=desc)
    while refs:
        done, refs = ray.wait(refs)
        if desc is not None:
            pbar.update(len(done))
        ray.get(done[0])
    if desc is not None:
        pbar.close()


def get(data):
    """get remote data"""
    if isinstance(data, (list, tuple)):
        dtype = type(data)
        ret = dtype(get(item) for item in data)
        return ret
    if isinstance(data, dict):
        return {key: get(value) for key, value in data.items()}
    while isinstance(data, ray.ObjectRef):
        data = ray.get(data)
    return data
