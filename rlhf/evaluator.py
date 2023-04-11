import math
import ray
from rlhf.environment import PPOEnv
from rlhf import utils
from ray.util.queue import Queue
from itertools import cycle

class Evaluator(PPOEnv):
    """
    evaluator
    """
    def __init__(self, args, models):
        self.args = args
        if not isinstance(models, list):
            models = [models]
        self.models = models
        self.batch_size = args.generation_batch_size
        self._dataset = []
        self.data_iter = None
        self._padding_config = {}
        self._name2models = {}
        self._model2funcs = {}
        self.merged_buffer = {}
        self.model2iter = {}


    def setup(self):
        refs = []
        for i, model_replica in enumerate(self.models[0].replicas):
            ref = model_replica.master._build_dataloader.remote(self._dataset[i], is_cycle=False)
            refs.append(ref)
        utils.get(refs)

        for model in self.models:
            config = ray.get(model.replicas[0].master.padding_config.remote())
            self._padding_config.update(config)
            self._name2models[model.name] = model

    def set_dataset(self, dataset):
        data_len = len(dataset)
        data_part_num = self.models[0].num_replica
        indices = utils.split_index(data_len, data_part_num)

        for i, (start, end) in enumerate(indices):
            data_part = dataset[start:end]
            self._dataset.append(data_part)


    @property
    def num_eval_iteration(self):
        return sum(math.ceil(len(data) / self.batch_size) for data in self._dataset)


    def register_func(self, model_name, func):
        self._model2funcs[model_name] = func

    
    def eval_step(self, data_queue, model_out_queues):
        in_queue = data_queue
        for model in self.models:
            if model.name in self._model2funcs:
                func_name = self._model2funcs[model.name]
            else:
                func_name = "eval_step"
            self.generate_step_one_model(model, in_queue, model_out_queues[model], func_name)
            in_queue = model_out_queues[model][0]

        out_queues = [model_out_queues[model][1] for model in self.models]
        return self.get_merged_data(out_queues, encode=False)


    def eval(self, queue):
        data_queue = Queue()
        num_batch = self.num_eval_iteration
        data_providers = cycle(iter(self.models[0].replicas))
        out_queues = {}
        for model in self.models:
            out_queues[model] = [Queue() for i in range(2)]
        for mb in range(num_batch):
            query = next(data_providers).master.next_batch.remote()
            data_queue.put(self.encode_data(mb, query))
            data = self.eval_step(data_queue, out_queues)
            queue.put(data)
