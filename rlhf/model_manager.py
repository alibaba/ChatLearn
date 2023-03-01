import os
from rlhf.dist_actor import DistActor, DistTorchActor
from rlhf.model_wrapper import RLHFTorchWrapper
from rlhf import dlc_utils


class ModelManager:

    def __init__(self, rlhf_models, resouce_manager, global_args):
        self.local_models = rlhf_models
        self.resouce_manager = resouce_manager
        self.remote_models = []
        self.env_args = global_args.env_args
        self.rlhf_args = global_args.rlhf_args
        self.converted = False
        self.free_ports = []
        if self.env_args.platform == "DLC":
            # port for DLC jobs, the first port is reserved for ray start
            self.free_ports = dlc_utils.get_free_ports()[1:]
        self.port_index = 0

    
    def remote(self) -> list:
        """
        convert model to remote
        """
        if self.converted:
            return self.remote_models
        for model in self.local_models:
            self.remote_models.append(self._to_dist_actor(model))
        self.converted = True
        return self.remote_models

    def get_free_port(self):
        port = self.free_ports[self.port_index]
        self.port_index += 1
        return port


    def _to_dist_actor(self, model) -> DistActor:
        """
        Convert one model to DistActor and place it to devices
        """
        num_replica = self.rlhf_args.num_rollout_worker if not model.trainable else 1
        model.set_num_replica(num_replica)

        placement_group = self.resouce_manager.get_placement_group(model)
        gpu_per_node = self.resouce_manager.gpu_per_node
        if isinstance(model, RLHFTorchWrapper):
            if self.env_args.platform == "DLC":
                free_port = self.get_free_port()
            else:
                free_port = None
            return DistTorchActor(model, placement_group, gpu_per_node, free_port).remote()
        return DistActor(model, placement_group, gpu_per_node).remote()

