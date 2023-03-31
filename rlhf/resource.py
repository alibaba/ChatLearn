import os
import ray
from ray.util.placement_group import placement_group


class ResourceManager:
    """
    Manage hardware resources for each task.
    """

    def __init__(self, models, colocation_groups=None):
        self.models = models
        self.name2models = {model.name: model for model in self.models}
        self.model_to_placegroup = {}
        resource = ray.nodes()[0]['Resources']
        self.gpu_per_node = int(resource['GPU'])
        self.cpu_per_node = int(resource['CPU'])

    def create_placement_group(self, model, strategy="PACK"):
        """
        create resource placement group given model device args
        """
        num_device = model.num_device
        if num_device <= self.gpu_per_node:
            cpu_count = int(self.cpu_per_node * num_device / self.gpu_per_node)
            bundles = [{"GPU": num_device, "CPU": cpu_count}]
        else:
            assert num_device % self.gpu_per_node == 0
            num_nodes = num_device // self.gpu_per_node
            bundles = [{"GPU": self.gpu_per_node, "CPU": self.cpu_per_node} for _ in range(num_nodes)]
        pg = placement_group(bundles, strategy=strategy)
        model.placement_group = pg
        return pg