from rlhf.dist_actor import DistActor, DistTorchActor
from rlhf.model_wrapper import RLHFTorchWrapper


class ModelManager:

    def __init__(self, rlhf_models, resouce_manager, rlhf_args):
        self.local_models = rlhf_models
        self.resouce_manager = resouce_manager
        self.remote_models = []
        self.rlhf_args = rlhf_args
        self.converted = False

    
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


    def _to_dist_actor(self, model) -> DistActor:
        """
        Convert one model to DistActor and place it to devices
        """
        num_replica = self.rlhf_args.num_rollout_worker if not model.trainable else 1
        model.set_num_replica(num_replica)

        placement_group = self.resouce_manager.get_placement_group(model)
        if isinstance(model, RLHFTorchWrapper):
            actor_cls = DistTorchActor
        else:
            actor_cls = DistActor
        gpu_per_node = self.resouce_manager.gpu_per_node
        return actor_cls(model, placement_group, gpu_per_node).remote()

