import ray
from rlhf.global_vars import get_args
from rlhf.arguments import parse_args_from_yaml
from rlhf.utils import get_free_port, get_host_addr
from rlhf import dlc_utils
from rlhf import utils
from rlhf.global_vars import set_global_variables
import ray.util.collective as col
import os
from rlhf.megatron_utils import build_pipeline_layer_name_mapping
from rlhf.logger import logger


class RLHFModule:

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
        self.args = args
        self.model_args = None
        if args.model_config_file:
            self.model_args = parse_args_from_yaml(args.model_config_file)
        self._num_replica = self.rlhf_args.num_rollout_worker if not self.trainable else 1
        assert self.num_replica >= 1
        self._param_ranks = None
        self._named_parameters = None
        self.error_signal = None
        self._rank = None
        self._world_size = None
        self._group_name = None
        self._dataloader = None
        self._has_next = True
        self._kl_coef = None
        self._padding_config = {}
        self._storage = None


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


    def error(self):
        ray.get(self.error_signal.set.remote())


    def setup(self):
        """
        init model env / create model / data
        """
        pass


    def validate(self):
        return "ok"


    def before_episode(self):
        """
        operations before one episode
        """
        pass
    

    def after_episode(self):
        """
        operations after one episode
        """
        pass


    def set_dataloader(self, dataloader):
        self._dataloader = dataloader
        self._data_iter = iter(self._dataloader)
        self._has_next = True


    def next_batch(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            self._has_next = False


    @property
    def num_replica(self):
        return self._num_replica


    def forward_step(self, data):
        # forward step
        pass


    def train_step(self, data, train_info):
        # train step
        # train_info includes training information, e.g., current iteration
        pass


    def save_checkpoint(self, iteration):
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


    def get_parameter_names(self, requires_grad=True):
        if requires_grad:
            names = [key for key, param in self.named_parameters.items() if param.requires_grad]
        else:
            names = [key for key in self.named_parameters]
        return names


    def get_parameter(self, name):
        if name not in self.named_parameters:
            raise Exception(f"parameter {name} not exits")
        return self.named_parameters[name]


    def exist_parameter(self, name):
        return name in self.named_parameters


    def parameter_shape(self, name):
        return self.get_parameter(name).shape


    def send_parameter(self, name, dst_rank, group_name):
        try:
            tensor = self.get_parameter(name)
            col.send(tensor, dst_rank, group_name)
        except Exception as e:
            self.error()
            raise


    def recv_parameter(self, name, src_rank, group_name):
        try:
            tensor = self.get_parameter(name)
            col.recv(tensor, src_rank, group_name)
        except Exception as e:
            self.error()
            raise

    
    def pipeline_model_parallel_size(self):
        pass


    def tensor_model_parallel_size(self):
        pass


    def num_layers(self):
        pass


    def set_storage(self, storage):
        self._storage = storage


    def put(self, key, data):
        self._storage.put.remote(key, data)


    def get(self, key):
        ref = self._storage.get.remote(key)
        return utils.get(ref)


    def add_padding_config(self, key, padding_value=0.0, padding_type="right"):
        self._padding_config[key] = {"padding_value": padding_value, "padding_type": padding_type}
    

    def padding_config(self):
        return self._padding_config


class RLHFTorchModule(RLHFModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_addr_port(self):
        """
        Get node address and port
        """
        if dlc_utils.in_dlc_env():
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


class RLHFMegatronModule(RLHFTorchModule):

    def validate(self):
        min_device = self.pipeline_model_parallel_size() * self.tensor_model_parallel_size()
        if self.num_device < min_device:
            self.error()
            raise RuntimeError(f"num_device {self.num_device} should be greater than" \
                f"pipe size {self.pipeline_model_parallel_size()}" \
                f"x tensor parallel size {self.tensor_model_parallel_size()}")

    @property
    def megatron_args(self):
        import megatron
        return megatron.get_args()


    def pipeline_model_parallel_size(self):
        return self.megatron_args.pipeline_model_parallel_size

    
    def tensor_model_parallel_size(self):
        return self.megatron_args.tensor_model_parallel_size


    def num_layers(self):
        return self.megatron_args.num_layers


    def build_pipeline_layer_name_mapping(self):
        from megatron import mpu
        layers_per_stage = self.num_layers() // self.pipeline_model_parallel_size()
        rank = mpu.get_pipeline_model_parallel_rank()
        logger.info(f"build mapping for rank {rank} =========")
        if isinstance(self.model, list):
            assert len(self.model) == 1
            model = self.model[0]
        else:
            model = self.model
        name_mapping = build_pipeline_layer_name_mapping(layers_per_stage, rank, model)
        return name_mapping


