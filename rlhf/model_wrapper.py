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
from rlhf.timer import Timers
from itertools import cycle
import torch



class RLHFModule:

    def __init__(self, name, args=None, rank=None):
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
        self.config_dir = args.config_dir
        self.model_args = args.model_args
        self._num_replica = self.rlhf_args.num_rollout_worker if not self.trainable else 1
        assert self._num_replica >= 1
        self._param_ranks = None
        self._named_parameters = None
        self.error_signal = None
        self._rank = rank
        self._world_size = None
        self._group_name = None
        self._dataloader = None
        self._kl_coef = None
        self._padding_config = {}
        self._storage = None
        self._timers = None



    def set_env(self):
        """
        set system env, private

        :meta private:
        """
        pass


    def set_error_signal(self, error_signal):
        """
        signal for handling errors

        :meta private:
        """
        self.error_signal = error_signal


    def error(self):
        """
        :meta private:
        """
        ray.get(self.error_signal.set.remote())


    def init(self):
        """
        init env
        """
        pass


    def setup(self):
        """
        create model / data
        """
        pass


    def forward_step(self, data):
        """
        perform forward step for one batch

        Args:
            data: data for forward_step, type is dict

        Returns:
            k/v dict
        """
        pass


    def train_step(self, data, train_info):
        """
        Perform train_step for one batch, including a list of micro-batches

        Args:
            data: data for train_step, type is a list of dict, each dict is a micro-batch
            train_info: includes training information, e.g., "iteration"

        """
        pass


    def eval_step(self, data):
        """
        Perform eval_step for one batch

        Args:
            data: data for eval_step, type is dict

        Returns:
            k/v dict
        """


    def save_checkpoint(self, iteration):
        """
        Save checkpoint given iteration.

        Args:
            iteration: current training iteration

        """
        pass



    def put(self, key, data):
        """
        Put the data to shared storage.

        Args:
            key: use key to put
            data: data to save
        """
        self._storage.put.remote(key, data)


    def get(self, key):
        """
        Get data from shared storage using key

        Args:
            key: use key to get
        """
        ref = self._storage.get.remote(key)
        return utils.get(ref)


    def validate(self):
        """
        :meta private:
        """
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


    def _build_dataloader(self, data, is_cycle=True):
        """
        build and set the dataloader for the model

        Args:
            data: a list of string

        :meta private:
        """
        dataloader = self.build_dataloader(data)
        self._dataloader = dataloader
        self._data_iter = iter(self._dataloader)
        if is_cycle:
            self._data_iter = cycle(self._data_iter)


    def build_dataloader(self, data):
        """
        build the dataloader for the model

        Args:
            data: a list of string
        """
        pass


    def next_batch(self):
        """
        :meta private:
        """
        return next(self._data_iter)


    @property
    def num_replica(self):
        return self._num_replica


    def setup_collective_group(self, rank, world_size, backend, group_name):
        """
        :meta private:
        """
        self._group_name = group_name
        self._world_size = world_size
        col.init_collective_group(
            world_size, rank, backend=backend, group_name=group_name)


    def destroy_collective_group(self):
        """
        :meta private:
        """
        col.destroy_collective_group(self._group_name)


    def set_param_ranks(self, param_ranks):
        """
        set the ranks for parameters of first replica
        """
        self._param_ranks = param_ranks


    def get_param_ranks(self):
        """
        :meta private:
        """
        return self._param_ranks


    @property
    def rank(self):
        """
        :meta private:
        """
        return self._rank


    @property
    def named_parameters(self):
        """
        :meta private:
        """
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
        """
        :meta private:
        """
        if requires_grad:
            names = [key for key, param in self.named_parameters.items() if param.requires_grad]
        else:
            names = [key for key in self.named_parameters]
        return names


    def get_parameter(self, name):
        """
        :meta private:
        """
        if name not in self.named_parameters:
            raise Exception(f"parameter {name} not exits")
        return self.named_parameters[name]


    def exist_parameter(self, name):
        """
        :meta private:
        """
        return name in self.named_parameters


    def parameter_shape(self, name):
        """
        :meta private:
        """
        return self.get_parameter(name).shape


    def send_parameter(self, name, dst_rank, group_name):
        """
        :meta private:
        """
        try:
            tensor = self.get_parameter(name)
            col.send(tensor, dst_rank, group_name)
        except Exception as e:
            self.error()
            raise


    def recv_parameter(self, name, src_rank, group_name):
        """
        :meta private:
        """
        try:
            tensor = self.get_parameter(name)
            col.recv(tensor, src_rank, group_name)
        except Exception as e:
            self.error()
            raise

    
    def pipeline_model_parallel_size(self):
        """
        :meta private:
        """
        pass


    def tensor_model_parallel_size(self):
        """
        :meta private:
        """
        pass


    def num_layers(self):
        """
        :meta private:
        """
        pass


    def set_storage(self, storage):
        """
        :meta private:
        """
        self._storage = storage


    def timers(self, name):
        """
        :meta private:
        """
        if self._timers is None:
            self._timers = Timers()
        return self._timers(name)


    def timer_summary(self):
        """
        :meta private:
        """
        if self._timers:
            return self._timers.log()


    def add_padding_config(self, key, padding_value=0.0, padding_type="right"):
        """
        Add spectial padding config for certain value

        Args:
            key: the key for data to be padded
            padding_value: padding value, default is 0
            padding_type: default right, can be right/left
        """
        self._padding_config[key] = {"padding_value": padding_value, "padding_type": padding_type}
    

    def padding_config(self):
        """
        :meta private:
        """
        return self._padding_config


    def peak_memory(self):
        """
        :meta private:
        """
        pass



class RLHFTorchModule(RLHFModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_addr_port(self):
        """
        Get node address and port

        :meta private:
        """
        if dlc_utils.in_dlc_env():
            addr = dlc_utils.get_addr()
            port = None
        else:
            addr = get_host_addr()
            port = get_free_port()
        return addr, port


    def get_visible_gpus(self):
        """
        :meta private:
        """
        return ray.get_gpu_ids()
        

    def set_env(self, args):
        """
        :meta private:
        """
        for key in ['RANK', 'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'LOCAL_RANK']:
            assert key in args, f"{key} is not set for RLHFTorchWrapper"
            os.environ[key] = str(args[key])
        return 1


    def get_dist_env(self):
        envs = {}
        for key in ['RANK', 'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'LOCAL_RANK']:
            envs[key] = os.environ[key]
        return envs


    def peak_memory(self):
        """
        :meta private:
        """
        return torch.cuda.max_memory_allocated() / (1024**3)


    @property
    def data_parallel_size(self):
        pass


    @property
    def data_parallel_rank(self):
        pass


class RLHFMegatronModule(RLHFTorchModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.trainable:
            # inference only
            self.model_args["micro_batch_size"] = self.rlhf_args.generation_batch_size
        else:
            self.model_args["micro_batch_size"] = self.rlhf_args.train_micro_batch_size
            self.model_args["global_batch_size"] = self.rlhf_args.train_global_batch_size


    def validate(self):
        """
        :meta private:
        """
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
        """
        get pipeline_model_parallel_size
        """
        return self.megatron_args.pipeline_model_parallel_size

    
    def tensor_model_parallel_size(self):
        """
        get tensor_model_parallel_size
        """
        return self.megatron_args.tensor_model_parallel_size


    @property
    def data_parallel_size(self):
        from megatron import mpu
        return mpu.get_data_parallel_world_size()


    @property
    def data_parallel_rank(self):
        from megatron import mpu
        return mpu.get_data_parallel_rank()


    def num_layers(self):
        """
        :meta private:
        """
        return self.megatron_args.num_layers


    def build_pipeline_layer_name_mapping(self):
        """
        :meta private:
        """
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

