import math
import ray
from rlhf.environment import PPOEnv
from rlhf import utils
from ray.util.queue import Queue
from itertools import cycle
from rlhf.model_wrapper import RLHFModule
from rlhf.global_vars import get_args
from tqdm import tqdm


class Evaluator(PPOEnv):
    """
    evaluator
    """
    def __init__(self, models, args=None):
        self._lazy_init = False
        self.args = args
        if not isinstance(models, list):
            models = [models]
        self.models = models
        self._dataset = []
        self.data_iter = None
        self._padding_config = {}
        self._name2models = {}
        self._model2funcs = {}
        self.merged_buffer = {}
        self.model2iter = {}
        self._original_dataset = None
        self._post_process_func = None


    @property
    def batch_size(self):
        return self.args.generation_batch_size


    def update_models(self, models):
        new_models = []
        name_to_new_models = {model.name:model for model in models}
        for model in self.models:
            new_models.append(name_to_new_models[model.name])
        self.models = new_models
        if self.args is None:
            self.args = get_args().rlhf_args


    def setup(self):
        if self._lazy_init:
            self.set_dataset(self._original_dataset)

        refs = []
        for i, model_replica in enumerate(self.models[0].replicas):
            ref = model_replica.master._build_dataloader.remote(self._dataset[i], is_eval=True)
            refs.append(ref)
        utils.get(refs)

        for model in self.models:
            config = ray.get(model.replicas[0].master.padding_config.remote())
            self._padding_config.update(config)
            self._name2models[model.name] = model


    def set_dataset(self, dataset):
        if isinstance(self.models[0], RLHFModule):
            self._original_dataset = dataset
            self._lazy_init = True
            return self
        data_len = len(dataset)
        data_part_num = self.models[0].num_replica
        indices = utils.split_index(data_len, data_part_num)

        for i, (start, end) in enumerate(indices):
            data_part = dataset[start:end]
            self._dataset.append(data_part)
        return self


    @property
    def num_eval_iteration(self):
        return sum(math.ceil(len(data) / self.batch_size) for data in self._dataset)


    def register_func(self, model_name, func):
        self._model2funcs[model_name] = func
        return self

    
    def eval_step(self, data_queue, model_out_queues, return_last=True):
        in_queue = data_queue
        for model in self.models:
            if model.name in self._model2funcs:
                func_name = self._model2funcs[model.name]
            else:
                func_name = "eval_step"
            self.generate_step_one_model(model, in_queue, model_out_queues[model], func_name)
            in_queue = model_out_queues[model][0]

        if return_last:
            out_queues = [model_out_queues[self.models[-1]][1]]
        else:
            out_queues = [model_out_queues[model][1] for model in self.models]
        return self.get_merged_data(out_queues, encode=False)


    def set_post_process_func(self, post_process_func):
        self._post_process_func = post_process_func
        return self


    def eval(self, ppo_iter=None, train_iteration=None, return_queue=False, return_last=True):
        queue = Queue()
        data_queue = Queue()
        num_batch = self.num_eval_iteration
        refs = []
        for model in self.models[0].replicas:
            refs.append(model.master.reset_eval_data_iter.remote())
        utils.get(refs)
        data_providers = cycle(iter(self.models[0].replicas))
        out_queues = {}

        for k, model in enumerate(self.models):
            if k < len(self.models) - 1:
                queue_num = 1
            else:
                queue_num = 2
            out_queues[model] = [Queue() for i in range(queue_num)]
        for mb in range(num_batch):
            query = next(data_providers).master.next_batch.remote(is_eval=True)
            data_queue.put(self.encode_data(mb, query))
            data = self.eval_step(data_queue, out_queues)
            queue.put(data)
        if return_queue:
            # end of evaluation
            queue.put(None)
            return queue
        else:
            results = []
            # last one is None
            total = queue.qsize()
            for i in tqdm(range(total), desc="evaluation"):
                res = utils.get(queue.get())
                if return_last:
                    res = res[0]
                results.append(res)
            if self._post_process_func is not None:
                eval_info = {}
                if ppo_iter is not None:
                    eval_info["episode_iteration"] = ppo_iter
                if train_iteration is not None:
                    eval_info["train_iteration"] = train_iteration
                self._post_process_func(results, eval_info)
            return results

