
import os
import time

from ray.util.queue import Queue

import torch

import chatlearn
from chatlearn.utils import future
from chatlearn.runtime.engine import BaseEngine
from chatlearn import TorchModule
from chatlearn.utils.timer import Timers
from chatlearn.utils.arguments import parse_args
from chatlearn.runtime.executor import Executor


def test_timers():
    timers = Timers()
    timers("aaa").start()
    time.sleep(1)
    sec = timers("aaa").elapsed()
    assert sec < 1.1 and sec > 1
    print(sec)
    time.sleep(1)
    sec = timers("aaa").elapsed()
    assert sec < 1.1 and sec > 1
    print(sec)

    timers("aaa").stop()
    timers("aaa").start()
    time.sleep(0.5)
    timers("aaa").stop()
    time.sleep(0.5)
    timers("aaa").start()
    time.sleep(0.5)
    timers("aaa").stop()

    sec = timers("aaa").elapsed()
    assert sec > 1 and sec < 1.1


def test_args():
    args0 = parse_args()
    assert args0.runtime_args.num_training_epoch == 3, args0.runtime_args.num_training_epoch
    assert args0.models["policy"].model_config_file == "configs/model.yaml", args0.models["policy"].model_config_file
    assert args0.models['reference'].gpu_per_process == 1
    assert args0.models['policy'].args_dict['generate_config']["num_beams"] == 1
    assert args0.runtime_args.get("unknown_args") == "test_unknown"
    assert args0.models['policy'].args_dict['model_config']['attention_probs_dropout_prob'] == 0.1
    assert args0.models['policy'].args_dict['test'] == 123
    assert args0.models['policy'].args_dict['generate_config']['eos_token_id'] == 103

def test_args_2():
    os.environ["num_training_epoch"] = "2"
    args0 = parse_args()
    assert args0.runtime_args.num_training_epoch == 2

def test_dist_actor():
    class PolicyModel(TorchModule):
        def forward_step(self, data, iteration=0):
            #assert data['a'].device.type == 'cpu', data['a'].device.type
            return data


    model = PolicyModel('policy')

    engine = BaseEngine(model)
    engine.setup()
    a = torch.ones([1])
    b = torch.ones([1])
    model = engine.models[0].replicas[0]
    res0 = model.forward_step({'a': a, 'b': b})
    res0 = future.get(res0)[0]
    res0 = model.forward_step({'a': a, 'b': b})
    res0 = future.get(res0)[0]
    assert res0['a'].device.type == 'cpu', res0['a'].device

    visible_devices = model.get_visible_gpus()
    visible_devices = future.get(visible_devices)
    assert visible_devices == [[0]], visible_devices

    engine.logging_summary()
    engine.stop()
    print(res0)


def test_align():
    queues = []
    for j in [1,2,4]:
        num_producers = 2 * j
        in_queue = Queue()
        for i in range(num_producers):
            item = {
                f"tensor_{j}": torch.rand(8//num_producers,4+i),
                f"list_{j}": [[1+k+i,2+k+i] for k in range(8//num_producers)]
            }
            in_queue.put(item)
        queues.append(in_queue)

    assert [ele.qsize() for ele in queues] == [2, 4, 8]

    out_queues = Executor.align_out_queues(queues, encode=False)

    assert [ele.qsize() for ele in out_queues] == [2, 2, 2]

    for out_queue in out_queues:
        kvs = out_queue.get()
        for key, value in kvs.items():
            if "tensor" in key:
                assert value.shape[0] == 4
            if "list" in key:
                assert len(value) == 4, f"{value}"

TEST_CASE = [test_align, test_timers, test_args, test_args_2, test_dist_actor]