import os
import ray
from ray.util.placement_group import placement_group

def get_cpu_count():
    return os.cpu_count()


class ResourceManager:
    """
    Manage hardware resources for each task.
    """

    def __init__(self, models):
        self.models = models
        self.model_to_placegroup = {}
        for model in self.models:
            pg = self.create_placement_group(model)
            self.model_to_placegroup[model] = pg
        self.gpu_per_node = ray.nodes()[0]['Resources']['GPU']

    def create_placement_group(self, model):
        """
        create resource placement group given model device args
        """
        placement_groups = []
        cpu_count = get_cpu_count()
        for i in range(model.num_replica):
            num_device = model.num_device
            if num_device <= self.gpu_per_node:
                bundles = [{"GPU": num_device, "CPU": cpu_count*num_device/self.gpu_per_node}]
            else:
                assert num_device % self.gpu_per_node == 0
                num_nodes = num_device // self.gpu_per_node
                bundles = [{"GPU": self.gpu_per_node, "CPU": cpu_count} for _ in range(num_nodes)]
            pg = placement_group(bundles, strategy="STRICT_PACK")
            placement_groups.append(pg)
        return placement_groups




    def get_placement_group(self, model):
        """
        place to certain group
        """
        return self.model_to_placegroup[model]
