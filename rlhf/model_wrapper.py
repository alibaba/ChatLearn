import ray
from rlhf.global_vars import get_args
from rlhf.arguments import parse_args_from_yaml
from rlhf.utils import get_free_port, get_host_addr
from rlhf import dlc_utils
import os

class RLHFModelWrapper:

    def __init__(self, name, args=None):
        self.name = name
        global_args = get_args() if args is None else args
        self.global_args = global_args
        args = global_args.models[name]
        self.num_device = args.num_device
        self.gpu_per_process = args.gpu_per_process
        self.trainable = args.trainable
        self._num_replica = 1
        self.args = args
        self.model_args = None
        if args.model_config_file:
            self.model_args = parse_args_from_yaml(args.model_config_file)
        assert self.num_replica >= 1

    def set_env(self):
        """
        set system env, private
        """
        pass

    def setup(self):
        """
        init model env / create model / data
        """
        pass

    def set_num_replica(self, num_replica):
        self._num_replica = num_replica

    @property
    def num_replica(self):
        return self._num_replica

    def next_batch(self):
        # get next batch of data
        pass

    def generate(self, data):
        # generate output from data
        pass

    def forward_step(self, data):
        # forward step
        pass

    def backward_step(self, data):
        # backward step
        pass

    def train_step(self, data):
        # train step
        pass

    def save_checkpoint(self):
        pass

    def update_parameters(self):
        """
        update parameters
        """
        pass


class RLHFTorchWrapper(RLHFModelWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_addr_port(self):
        """
        Get node address and port
        """
        if self.global_args.env_args.platform == "DLC":
            addr = dlc_utils.get_addr()
            port = None
        else:
            addr = get_host_addr()
            port = get_free_port()
        return addr, port


    def get_visible_gpus(self):
        return ray.get_gpu_ids()
        

    def set_env(self, args):
        for key in ['RANK', 'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE']:
            assert key in args, f"{key} is not set for RLHFTorchWrapper"
            os.environ[key] = str(args[key])
        return 1

