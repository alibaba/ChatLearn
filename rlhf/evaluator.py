import math
import ray


class Evaluator:
    """
    evaluator
    """
    def __init__(self, args, policy, reward, index):
        self.index = index
        self.policy = policy
        self.reward = reward
        self.batch_size = args.generation_batch_size
        self._dataset = None
        self.data_iter = None
        self._padding_config = {}
        self._name2models = {}
        self._model2funcs = {}


    def setup(self):
        ref = self.policy.master._build_dataloader.remote(self._dataset, is_cycle=False)
        ray.get(ref)

        for model in [self.policy, self.reward]:
            config = ray.get(model.master.padding_config.remote())
            self._padding_config.update(config)
            self._name2models[model.name] = model

    def set_dataset(self, dataset):
        self._dataset = dataset


    @property
    def num_eval_iteration(self):
        return math.ceil(len(self._dataset) / self.batch_size)


    def register_func(self, model_name, func):
        self._model2funcs[model_name] = func

    
    def eval_step(self, query):
        if self.policy.name in self._model2funcs:
            policy_func = getattr(self.policy, self._model2funcs[self.policy.name])
        else:
            policy_func = self.policy.eval_step
        policy_output = policy_func(query)
        if self.reward.name in self._model2funcs:
            reward_func = getattr(self.reward, self._model2funcs[self.reward.name])
        else:
            reward_func = self.reward.eval_step
        reward_outputs = reward_func(policy_output[0])
        return policy_output[0], reward_outputs[0]


    def eval(self, queue):
        num_batch = self.num_eval_iteration
        for i in range(num_batch):
            query = self.policy.master.next_batch.remote()
            data = self.eval_step(query)
            queue.put(data)
