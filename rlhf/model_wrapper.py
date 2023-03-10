import ray
from rlhf.global_vars import get_args
from rlhf.arguments import parse_args_from_yaml
from rlhf.utils import get_free_port, get_host_addr
from rlhf import dlc_utils
from rlhf.global_vars import set_global_variables
import ray.util.collective as col
import os

class RLHFModelWrapper:

    def __init__(self, name, args=None):
        self.name = name
        if args is None:
            global_args = get_args()
        else:
            global_args = args
            set_global_variables(args)
        self.global_args = global_args
        args = global_args.models[name]
        self.num_device = args.num_device
        self.gpu_per_process = args.gpu_per_process
        self.trainable = args.trainable
        self.rlhf_args = self.global_args.rlhf_args
        self._num_replica = 1
        self.args = args
        self.model_args = None
        if args.model_config_file:
            self.model_args = parse_args_from_yaml(args.model_config_file)
        assert self.num_replica >= 1
        self._param_ranks = None
        self._named_parameters = None
        self.error_signal = None
        self._rank = None
        self._world_size = None
        self._group_name = None


    def set_env(self):
        """
        set system env, private
        """
        pass


    def set_error_signal(self, error_signal):
        """
        signal for handling errors
        """
        self.error_signal = error_signal


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


    def forward_step(self, data):
        # forward step
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


    def setup_collective_group(self, rank, world_size, backend, group_name):
        self._rank = rank
        self._group_name = group_name
        self._world_size = world_size
        col.init_collective_group(
            world_size, rank, backend=backend, group_name=group_name)


    def destroy_collective_group(self):
        col.destroy_collective_group(self._group_name)


    def set_param_ranks(self, param_ranks):
        self._param_ranks = param_ranks


    def get_param_ranks(self):
        return self._param_ranks


    @property
    def named_parameters(self):
        if self._named_parameters is None:
            if not isinstance(self.model, list):
                model = [self.model]
            else:
                model = self.model
            self._named_parameters = {}
            for partition in model:
                for item in partition.named_parameters():
                    self._named_parameters[item[0]] = item[1]
        return self._named_parameters


    def get_parameter_names(self):
        names = [key for key in self.named_parameters]
        return names


    def get_parameter(self, name):
        if name not in self.named_parameters:
            raise Exception(f"parameter {name} not exits")
        return self.named_parameters[name]

    def exist_parameter(self, name):
        return name in self.named_parameters


    def send_parameter(self, name, dst_rank, group_name):
        try:
            tensor = self.get_parameter(name)
            col.send(tensor, dst_rank, group_name)
        except Exception as e:
            ray.get(self.error_signal.set.remote())
            raise


    def recv_parameter(self, name, src_rank, group_name):
        try:
            tensor = self.get_parameter(name)
            col.recv(tensor, src_rank, group_name)
        except Exception as e:
            ray.get(self.error_signal.set.remote())
            raise


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

