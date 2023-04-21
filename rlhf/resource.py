import ray
import ray.experimental.state.api
from rlhf.logger import logger
from ray.util.placement_group import placement_group
import time


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


    def get_placement_group_state(self, pg):
        try:
            state = ray.experimental.state.api.get_placement_group(pg.id.hex())["state"]
            return state
        except Exception as e:
            logger.warn(f"fail to get placement_group state {e}")


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
        warn_once = True
        while self.get_placement_group_state(pg) == "PENDING":
            if warn_once:
                logger.info(ray.experimental.state.api.list_nodes())
                logger.info(f"waiting for placement group to be created for {model.name} {pg.bundle_specs}")
                warn_once = False
            time.sleep(1)
        logger.info(f"create placement_group {pg.bundle_specs} for model {model.name} done")
        return pg
