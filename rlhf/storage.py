import ray
from rlhf.logger import logger

@ray.remote
class Storage:

    def __init__(self):
        self._storage = {}


    def put(self, key, data):
        ref = ray.put(data)
        self._storage[key] = ref


    def get(self, key):
        ref = self._storage.get(key)
        if ref is None:
            logger.warn(f"{key} is not found in storage")
            return None
        return ray.get(ref)

