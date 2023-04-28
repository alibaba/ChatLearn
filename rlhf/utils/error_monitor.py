import time

import ray

from rlhf.utils.logger import logger


@ray.remote
class ErrorMonitor:
    def __init__(self, error_signal, remote_models, group_names):
        self.error_signal = error_signal
        self.remote_models = remote_models
        self.collective_groups = group_names



    def monitor(self):
        while True:
            catch_err = ray.get(self.error_signal.is_set.remote())
            if catch_err:
                break
            time.sleep(2)
        logger.exception("error found")
        for group_name in self.collective_groups:
            col.destroy_collective_group(group_name)
        for model in self.remote_models:
            model.terminate()
        try:
            exit_actor = ray.get_actor("ExitActor")
            ray.kill(exit_actor)
        except Exception as e:
            pass
        ray.shutdown()


@ray.remote(num_cpus=0)
class ErrorSignalActor:
    def __init__(self):
        self.error_state = False

    def set(self):
        self.error_state = True

    def is_set(self):
        return self.error_state
