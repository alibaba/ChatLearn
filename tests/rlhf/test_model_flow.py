import chatlearn
from chatlearn import TorchModule
from chatlearn.runtime.executor import Executor
from chatlearn.runtime.model_flow import ModelFlow
from chatlearn.runtime.dist_actor import DistModel

from utils import CustomDataset


class PolicyModel(TorchModule):

    def forward_step(self, data, iteration):
        return data

    def forward_step2(self, data, iteration):
        return data

    def forward_step3(self, data, iteration):
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset


class ReferenceModel(TorchModule):

    def forward_step(self, data, iteration):
        return data


class RewardModel(TorchModule):

    def forward_step(self, data, iteration):
        return data


class ValueModel(TorchModule):

    def forward_step(self, data, iteration):
        return data
    
    def forward_step2(self, data, iteration):
        return data


class PPOPolicy(TorchModule):

    def train_step(self, data, iteration):
        return num_mb

class PPOValue(TorchModule):

    def train_step(self, data, iteration):
        return num_mb

class MockDistActor:

    def __init__(self, model):
        self.model = model
        self.name = model.name

def test_model_flow():
    policy = PolicyModel("policy")
    reference = ReferenceModel("reference")
    reward = RewardModel("reward")
    value = ValueModel("value")

    def env_compute_flow(batch):
        policy_out = policy.forward_step(batch)
        ref_out = reference.forward_step(policy_out)
        policy_out2 = policy.forward_step2(ref_out)
        value_out = value.forward_step2(policy_out2)
        policy_out3 = policy.forward_step3(value_out)
        reward_out = reward.forward_step(policy_out3, policy_out, ref_out, value_out)
        return value_out, reward_out

    env = Executor(env_compute_flow)
    model_flow = ModelFlow(env)
    models = [policy, reference, reward, value]
    mock_dist_models = []

    for model in models:
        dist_model = DistModel()
        dist_model.add_replica(MockDistActor(model))
        mock_dist_models.append(dist_model)

    model_flow.trace(mock_dist_models, env_compute_flow)

    assert model_flow.model_nodes[0].name == 'policy'
    assert model_flow.model_nodes[0].func_name == "forward_step"
    assert model_flow.model_nodes[1].name == 'reference'
    assert model_flow.model_nodes[1].func_name == "forward_step"
    assert model_flow.model_nodes[2].name == 'policy'
    assert model_flow.model_nodes[2].func_name == "forward_step2"
    assert model_flow.model_nodes[3].name == 'value'
    assert model_flow.model_nodes[3].func_name == "forward_step2"
    assert model_flow.model_nodes[4].name == 'policy'
    assert model_flow.model_nodes[4].func_name == "forward_step3"
    assert model_flow.model_nodes[5].name == 'reward'
    assert model_flow.model_nodes[5].func_name == "forward_step"

    def env_compute_flow(batch):
        policy_out = policy.forward_step(batch)
        ref_out = reference.forward_step(policy_out)
        value_out = value.forward_step(policy_out)
        reward_out = reward.forward_step(policy_out, ref_out, value_out)
        return value_out, reward_out

    env = Executor(env_compute_flow)
    model_flow = ModelFlow(env)
    models = [policy, reference, reward, value]
    mock_dist_models = []

    for model in models:
        dist_model = DistModel()
        dist_model.add_replica(MockDistActor(model))
        mock_dist_models.append(dist_model)

    model_flow.trace(mock_dist_models, env_compute_flow)

    assert model_flow.model_nodes[0].name == 'policy'
    assert model_flow.model_nodes[0].func_name == "forward_step"
    assert model_flow.model_nodes[1].name in ['reference', "value"]
    assert model_flow.model_nodes[1].func_name == "forward_step"
    assert model_flow.model_nodes[2].name in ['reference', "value"]
    assert model_flow.model_nodes[2].func_name == "forward_step"
    assert model_flow.model_nodes[3].name == 'reward'
    assert model_flow.model_nodes[3].func_name == "forward_step"

    def env_compute_flow(batch):
        policy_out = policy.forward_step(batch)
        ref_out = reference.forward_step(batch)
        return policy_out, ref_out

    env = Executor(env_compute_flow)
    model_flow = ModelFlow(env)
    models = [policy, reference]
    mock_dist_models = []

    for model in models:
        dist_model = DistModel()
        dist_model.add_replica(MockDistActor(model))
        mock_dist_models.append(dist_model)

    model_flow.trace(mock_dist_models, env_compute_flow)

    assert model_flow.model_nodes[0].name in ['policy', 'reference']
    assert model_flow.model_nodes[0].func_name == "forward_step"
    assert model_flow.model_nodes[1].name in ['policy', 'reference']
    assert model_flow.model_nodes[1].func_name == "forward_step"
    

TEST_CASE = [test_model_flow]