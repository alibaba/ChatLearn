import copy
import logging
import traceback
import time

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


class TestCaseRunner():
    def __init__(self):
        self.passed = []
        self.failed = []
        # We want enable debug logging for Test
        self._set_log_level_debug()

    def _set_log_level_debug(self, level=logging.DEBUG):
        # 获取根Logger并设置级别
        logger = logging.getLogger("ChatLearn")
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
    def run_test_case(self, case):
        try:
            print(f"\033[1;33m=========Test Case {case.__name__} starting =========\033[0m")
            case()
            print(f"\033[1;32m=========Test Case {case.__name__} passed =========\033[0m")

            self.passed.append(case.__name__)
        except Exception as e:
            print (f"Caught an Execption Error {e}")
            traceback.print_exc()
            print(f"\033[1;31m=========Test Case {case.__name__} failed =========\033[0m")
            self.failed.append(case.__name__)
    
    def report(self, seconds_used):
        total = len(self.passed) + len(self.failed)
        print(f"\033[1;32m=========Test Case Time:  {seconds_used} seconds=========\033[0m")
        print(f"\033[1;32m=========Test Case Total:  {total}=========\033[0m")
        print(f"\033[1;32m=========Test Case Passed: {len(self.passed)}=========\033[0m")
        for case in self.passed:
            print(f"\033[1;32mTest Case {case} passed\033[0m")

        if len(self.failed) == 0:
            print(f"\033[1;32m=========Test Case Passed Rate: 100% =========\033[0m")
            return 0

        print(f"\033[1;31m=========Test Case Failed: {len(self.failed)}=========\033[0m")
        for case in self.failed:
            print(f"\033[1;31mTest Case {case} failed!\033[0m")

        result = "{:.2f} %".format(len(self.passed) * 100 / total)
        print(f"\033[1;31m=========Test Case Passed Rate: {result} =========\033[0m")
        return 1

def run_test(cases, name = None):
    # run all cases if not name, or run name only
    runner = TestCaseRunner()
    ts = time.time()
    for case in cases:
        if name is not None:
            if case.__name__ == name:
                runner.run_test_case(case)
                break
        else:
            runner.run_test_case(case)
    te = time.time()
    seconds_used = "{:.2f}".format(te - ts)
    return runner.report(seconds_used)