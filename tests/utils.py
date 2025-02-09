import copy

from torch.utils.data import Dataset

from chatlearn import TorchModule
from chatlearn.utils import future


class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.collate_fn = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"query": self.data[idx]}


class PolicyModel(TorchModule):

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        query = data["query"]
        data["policy_out"] = query
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset


class ReferenceModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reference forward =========", flush=True)
        query = data["policy_out"].cuda()
        data["ref_out"] = query
        return data


class RewardModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reward forward =========", flush=True)
        data["reward_out"] = data["ref_out"].cuda() + data["policy_out"].cuda()
        return data


class ValueModel(TorchModule):

    def forward_step(self, data, iteration):
        print("value forward =========", flush=True)
        data["value_out"] = data["policy_out"].cuda() * 3
        return data


class PPOPolicy(TorchModule):

    def train_step(self, data, iteration):
        print("ppo policy train_step =========", flush=True)
        num_mb = len(data)
        return num_mb


class PPOValue(TorchModule):

    def train_step(self, data, iteration):
        print("ppo value train_step =========", flush=True)
        num_mb = len(data)
        return num_mb


def listdict_to_dictlist(ld, list_extend=True):
    '''
    [{k1: v11, k2: v2}, {k1: v12, k2: v2},....] => {k1: [v11, v12..], k2: [v21, v22...]}
    if v11 is list then k1: v11 + v12
    :param ld:
    :return:
    '''
    res = copy.deepcopy(ld[0])
    for res_key, v in res.items():
        if list_extend and isinstance(res[res_key], list):
            continue

        res[res_key] = [v]

    for d in ld[1:]:
        for key, v in d.items():
            if list_extend and isinstance(d[key], list):
                res[key].extend(v)
            else:
                res[key].append(v)

    return res

def assert_consumed_samples(engine, model_names, ground_truth:int):
    for model_name in model_names:
        assert future.get(engine.named_models[model_name].replicas[0].get_runtime_args()[0]).consumed_samples == ground_truth, (
            f"model {model_name} consumed "
            f"{future.get(engine.named_models[model_name].replicas[0].get_runtime_args()[0]).consumed_samples} samples, "
            f"while ground_truth = {ground_truth}, "
        )
