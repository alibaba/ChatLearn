import torch
from torch.utils.data import Dataset

import chatlearn
from chatlearn import Engine
from chatlearn import TorchModule
from chatlearn import RLHFEngine

from chatlearn.runtime.environment import Environment
from chatlearn.runtime.trainer import Trainer

from utils import CustomDataset, RewardModel, ValueModel, PPOPolicy, PPOValue


class PolicyModel(TorchModule):

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        query = data["query"]
        bs = query.size(0)
        data["policy_out"] = torch.ones([bs, 1024]).cuda()
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset


class ReferenceModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reference forward =========", flush=True)
        query = data["policy_out"].cuda()
        data["ref_out"] = query * 2
        return data

class EvalEngine(Engine):
    def __init__(self, eval_flow=None, evaluator=None):
        if evaluator is None:
            evaluator = Evaluator(eval_flow)
        super().__init__(evaluator=evaluator)
    
    def setup(self):
        super().setup()
        self.evaluator.set_multiple_datasets(self._all_datasets)
        self.evaluator.set_timers(self.timers)

    def eval(self, cur_iter=None, train_iteration=None):
        """
        Start evaluating.
        """
        self.setup()
        self.evaluator.setup()
        self.timers("episode").start()
        results = self.evaluator.eval(
            cur_iter=cur_iter, train_iteration=train_iteration)
        self.timers("episode").stop()
        return results

def test_rlhf_custom():
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    def env_compute_flow(batch):
        policy_out = policy.forward_step(batch)
        ref_out = reference.forward_step(policy_out)
        value_out = value.forward_step(policy_out)
        reward_out = reward.forward_step(policy_out, ref_out, value_out)
        return value_out, reward_out

    def trainer_compute_flow(batch):
        ppo_policy.train_step(batch)
        ppo_value.train_step(batch)

    env = Environment(env_compute_flow)
    trainer = Trainer(trainer_compute_flow)

    engine = Engine(env, trainer)
    engine.set_parameter_sync(ppo_policy, policy)
    engine.set_parameter_sync(ppo_value, value)
    assert policy.num_replica == 1
    assert reference.num_replica == 1
    data = torch.ones([1024])
    engine.set_dataset([data] * 35)

    engine.learn()
    assert len(engine.env._all_datasets[0]) == 35, len(engine.env._all_datasets[0])
    engine.stop()

def test_rlhf_cpu():
    chatlearn.get_args().models["policy"].num_replica = 2
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")
    ppo_policy = PPOPolicy("ppo_policy")
    ppo_value = PPOValue("ppo_value")

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
    #assert policy.num_replica == 2

    data = torch.ones([1024])
    engine.set_dataset([data] * 35)
    engine.learn()
    if policy.num_replica == 2:
        assert reference.num_replica == 1
        data = torch.ones([1024])
        assert len(engine.env._all_datasets[0]) == 35, len(engine.env._all_datasets[0])
        visible_devices = engine.models[0].replicas[0].get_visible_gpus()
        visible_devices = future.get(visible_devices)
        assert visible_devices == [[0]], visible_devices
        visible_devices = engine.models[0].replicas[1].get_visible_gpus()
        visible_devices = future.get(visible_devices)
        assert visible_devices == [[1]], visible_devices
    engine.stop()


def test_eval_data_input():
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")

    def env_compute_flow(batch):
        policy_out = policy.forward_step(batch)
        ref_out = reference.forward_step(policy_out, batch)
        return ref_out

    assert policy.num_replica == 1
    assert reference.num_replica == 1
    engine = EvalEngine(env_compute_flow)
    data = torch.ones([1024])
    engine.set_dataset([data] * 35)

    engine.eval()
    assert len(engine._all_datasets[0]) == 35, len(engine._all_datasets[0])
    engine.stop()

TEST_CASE = [test_rlhf_custom, test_rlhf_cpu, test_eval_data_input]
#TODO breaked from some reason, need to be fixed
TEST_CASE = [ ]