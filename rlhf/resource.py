import os
import ray
from ray.util.placement_group import placement_group


class ResourceManager:
    """
    Manage hardware resources for each task.
    """

    def __init__(self, models):
        self.models = models
        self.model_to_placegroup = {}
        resource = ray.nodes()[0]['Resources']
        self.gpu_per_node = int(resource['GPU'])
        self.cpu_per_node = int(resource['CPU'])
        for model in self.models:
            pg = self.create_placement_group(model)
            self.model_to_placegroup[model] = pg


    def create_placement_group(self, model, strategy="PACK"):
        """
        create resource placement group given model device args
        """
        placement_groups = []
        for i in range(model.num_replica):
            num_device = model.num_device
            if num_device <= self.gpu_per_node:
                cpu_count = int(self.cpu_per_node * num_device / self.gpu_per_node)
                bundles = [{"GPU": num_device, "CPU": cpu_count}]
            else:
                assert num_device % self.gpu_per_node == 0
                num_nodes = num_device // self.gpu_per_node
                bundles = [{"GPU": self.gpu_per_node, "CPU": self.cpu_per_node} for _ in range(num_nodes)]
            pg = placement_group(bundles, strategy=strategy)
            placement_groups.append(pg)
        return placement_groups


    def get_placement_group(self, model):
        """
        place to certain group
        """
        return self.model_to_placegroup[model]
