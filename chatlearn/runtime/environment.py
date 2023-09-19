# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Environment"""

import math
from itertools import cycle

from ray.util.queue import Queue

from chatlearn.utils import future
from chatlearn.utils import utils
from chatlearn.utils.logger import logger


class BaseEnv:
    """BaseEnv"""
    def __init__(self, args):
        self.args = args


class PPOEnv(BaseEnv):
    """
    PPO environment
    """

    def __init__(self, args, policy, reference, reward, value):
        super().__init__(args)
        self.sample_per_episode = args.sample_per_episode
        self.models = [policy, reference, reward, value]
        self.policy = policy
        self.reference = reference
        self.reward = reward
        self.value = value
        self.remote_models = [policy, reference, reward, value]
        assert self.sample_per_episode % len(self.policy.replicas) == 0, \
            "replica number of policy model must be divisible by sample_per_episode"
        self.sample_per_episode_per_replica = self.sample_per_episode // len(self.policy.replicas)
        self.batch_size = self.policy.module_args.generation_batch_size
        self.batch_per_episode = len(self.policy.replicas) \
            * math.ceil(self.sample_per_episode_per_replica / self.batch_size)
        self._dataset = None
        self.data_iter = None
        self._padding_config = {}
        self.merged_buffer = {}
        self.model2iter = {}
        self.model2group = {}
        self.reference_value_colocate = []
        self.num_reference_value_to_process = 1
        self.reference_value_results = []

    def setup(self, model_packs=None):
        assert isinstance(model_packs, list), \
            f"model_packs for PPOEnv must be a list, but got {type(model_packs)}"
        refs = []
        for i, policy_replica in enumerate(self.policy.replicas):
            ref = policy_replica.master._build_dataloader.remote(self._dataset[i],
                                                                 self.sample_per_episode_per_replica)
            refs.append(ref)
        future.get(refs)
        logger.info("set dataset for policy")

        for dist_model in [self.policy, self.reference, self.reward, self.value]:
            model = dist_model.replicas[0]
            config = future.get(model.master.padding_config.remote())
            self._padding_config.update(config)
        model_names = [m.name for m in self.remote_models]

        for group in self.args.colocation:
            new_group = []
            for model in group:
                if model in model_names:
                    new_group.append(model)
                    self.model2group[model] = new_group
        for model_pack in model_packs:
            model_name_pack = [model.name for model in model_pack]
            if len(model_name_pack) > 1 and self.reference.name in model_name_pack and self.value.name in model_name_pack:
                self.reference_value_colocate = [self.reference.name, self.value.name]
                self.num_reference_value_to_process = 2
                break

    def batch_generation_ranking(self, in_data):
        def sort_fun(ele):
            chinese = ""
            others = ""
            for s in ele:
                if '\u4e00' <= s <= '\u9fa5':
                    chinese += s
                else:
                    others += s
            return len(others.split(" ")) + len(chinese)
        for episode in range(self.episode_per_epoch):
            start = episode * self.sample_per_episode
            if episode < self.episode_per_epoch - 1:
                end = start + self.sample_per_episode
            else:
                end = len(in_data)
            cur_episode_sample = in_data[start:end]
            cur_episode_sample.sort(key=sort_fun, reverse=True)
            in_data = in_data[:start] + cur_episode_sample + in_data[end:]
        return in_data

    def set_dataset(self, dataset, drop_last=False, wrap_data=True):
        assert not drop_last or not wrap_data, "drop_last and wrap_data cannot be True at the same time"
        self._dataset = []
        # TODO: compare with use only master dataloader
        data_len = len(dataset)
        data_part_num = self.policy.num_replica
        if self.models[0].module_args.batch_generation.ranking:
            self._dataset = [[] for _ in range(data_part_num)]
            drop_len = data_len % self.batch_size
            if drop_len:
                if drop_last:
                    dataset = dataset[:-drop_len]
                    data_len -= drop_len
                elif wrap_data:
                    wrap_len = self.batch_size - drop_len
                    dataset = dataset + dataset[:wrap_len]
                    data_len += wrap_len
                if drop_last or wrap_data:
                    assert len(dataset) % self.batch_size == 0
            self.episode_per_epoch = math.ceil(data_len / self.sample_per_episode)
            logger.info("calling batch_generation_ranking")
            dataset = self.batch_generation_ranking(dataset)
            num_batch_per_episode = math.ceil(self.sample_per_episode / self.batch_size)
            for episode in range(self.episode_per_epoch):
                for batch in range(num_batch_per_episode):
                    start = episode * self.sample_per_episode + batch * self.batch_size
                    end = min(start + self.batch_size, self.sample_per_episode * (episode + 1))
                    end = min(end, data_len)
                    if start >= end:
                        break
                    self._dataset[int(batch % data_part_num)] += dataset[start:end]
            data_splits = [len(self._dataset[p_idx]) for p_idx in range(data_part_num)]
            assert data_len == sum(data_splits), \
                "Expect length of the whole dataset equals to sum length of all data splits."
        else:
            indices = utils.split_index(data_len, data_part_num)
            for start, end in indices:
                data_part = dataset[start:end]
                drop_len = len(data_part) % self.batch_size
                if drop_len:
                    if drop_last:
                        data_part = data_part[:-drop_len]
                    elif wrap_data:
                        wrap_len = self.batch_size - drop_len
                        data_part = data_part + data_part[:wrap_len]
                    if drop_last or wrap_data:
                        assert len(data_part) % self.batch_size == 0
                self._dataset.append(data_part)


    def build_data_loader(self):
        """generate prompts data loader"""
        # TODO: temply comment out, use dataloader from policy, consider later
        # return RLHFDataLoader(self._dataset, self.batch_size)

    def encode_data(self, mb, data):
        return {"env_iter": mb, "data": data}

    def decode_data(self, data):
        mb = data["env_iter"]
        data = data["data"]
        return mb, data

    def get_merged_data(self, queues, encode=True):
        queue0 = queues[0]

        mb0, data0 = self.decode_data(queue0.get())
        if isinstance(data0, list):
            # if model has multiple actors, just use the last one
            data0 = data0[-1]
        data_list = [data0]
        for index, queue in enumerate(queues[1:]):
            if index not in self.merged_buffer:
                self.merged_buffer[index] = {}
            if mb0 in self.merged_buffer[index]:
                data_list.append(self.merged_buffer[index].pop(mb0))
                continue
            while True:
                encoded_data = queue.get()
                mb, data = self.decode_data(encoded_data)
                if isinstance(data, list):
                    data = data[-1]
                if mb == mb0:
                    data_list.append(data)
                    break
                self.merged_buffer[index][mb] = data
        if encode:
            return self.encode_data(mb0, data_list)
        return data_list

    def get_all_merged_data(self, queues, out_queue, encode=True):
        queue0 = queues[0]
        while queue0.qsize() > 0:
            res = self.get_merged_data(queues, encode)
            out_queue.put(res)

    def _get_model(self, model):
        if len(model.replicas) == 1:
            return model.replicas[0]
        if model not in self.model2iter:
            self.model2iter[model] = cycle(iter(model.replicas))
        return next(self.model2iter[model])

    def generate_step_one_model(self, model, in_queue, out_queue, func_name="forward_step", to_empty_cache=None):
        """
        Args:
            model: DistModel
            in_queue: Queue
            out_queue: Queue
            func_name: str
            to_empty_cache: None or boolean
        """
        replica = self._get_model(model)

        if isinstance(in_queue, list):
            data = self.get_merged_data(in_queue)
        else:
            data = in_queue.get()
        mb, query = self.decode_data(data)
        func = getattr(replica, func_name)
        if isinstance(query, list):
            if to_empty_cache is None:
                output = func(*query)
            else:
                output = func(*query, to_empty_cache=to_empty_cache)
        else:
            if to_empty_cache is None:
                output = func(query)
            else:
                output = func(query, to_empty_cache=to_empty_cache)

        # If tp > 1 or pp > 1 for current model, its `output` will be a list whose
        #   length is the number of Actors. In this case, all members in the list
        #   are the same, and we choose output[-1] to put into out_queue.
        last_output = output[-1] if isinstance(output, list) else output
        if isinstance(out_queue, list):
            for oq in out_queue:
                oq.put(self.encode_data(mb, last_output))
        else:
            out_queue.put(self.encode_data(mb, last_output))
        # To ensure all Actors are finished synchronously, `output` itself should be returned
        return out_queue, output

    def generate_loop_one_model(self, model, in_queue, out_queue, func_name, num_batch):
        results = []
        replica_num = len(model.replicas)
        last_step_start = max(num_batch - replica_num, 0)
        for step in range(num_batch):
            if step >= last_step_start:
                _, data = self.generate_step_one_model(model, in_queue, out_queue, func_name, True)
            else:
                _, data = self.generate_step_one_model(model, in_queue, out_queue, func_name, False)
            results.append(data)

        # serialize model generation here if colocation detected
        # NOTE: ref and value can be parallelized if they are not colocated, e.g.: ref=4, value=4 and total_devices=8
        if model.name in self.model2group and len(self.model2group[model.name]) > 1 and model.name not in self.reference_value_colocate:
            future.wait(results, f"{model.name} {func_name}")
        if model.name in self.reference_value_colocate:
            if self.num_reference_value_to_process == 1:
                self.reference_value_results.extend(results)
                future.wait(self.reference_value_results, f"{self.reference_value_colocate} {func_name}")
                self.reference_value_results = []
                self.num_reference_value_to_process = 2
            else:
                assert self.num_reference_value_to_process == 2, \
                    "env.num_reference_value_to_process should be 2 if reference and value are not colocated, " \
                    f"but got {self.num_reference_value_to_process}."
                self.reference_value_results = results
                self.num_reference_value_to_process = 1


    def generate_step(self, data_queue, policy_out_queue, ref_out_queue, old_value_out_queue, reward_out_queue):
        # TODO: generate data_flow by ast parser
        self.generate_step_one_model(self.policy, data_queue, policy_out_queue, to_empty_cache=False)
        self.generate_step_one_model(self.reference, policy_out_queue[0], ref_out_queue, to_empty_cache=False)
        self.generate_step_one_model(self.value, policy_out_queue[1], old_value_out_queue, to_empty_cache=False)
        self.generate_step_one_model(self.reward, [policy_out_queue[2], ref_out_queue[0], old_value_out_queue],
                                     reward_out_queue, to_empty_cache=False)
        data = []
        if self.policy.module_args.return_rlhf_data:
            data.append(policy_out_queue[3])
        if self.reference.module_args.return_rlhf_data:
            data.append(ref_out_queue[1])
        if self.reward.module_args.return_rlhf_data:
            data.append(reward_out_queue)
        return self.get_merged_data(data, encode=False)

    def generate_loop_sync(self, data_queue, policy_out_queue, ref_out_queue, old_value_out_queue, reward_out_queue,
                           out_queue, num_batch):
        # TODO: generate data_flow by ast parser
        func_name = "forward_step"
        self.generate_loop_one_model(self.policy, data_queue, policy_out_queue, func_name, num_batch)
        self.generate_loop_one_model(self.reference, policy_out_queue[0], ref_out_queue, func_name, num_batch)
        self.generate_loop_one_model(self.value, policy_out_queue[1], old_value_out_queue, func_name, num_batch)
        self.generate_loop_one_model(self.reward, [policy_out_queue[2], ref_out_queue[0], old_value_out_queue],
                                     reward_out_queue, func_name, num_batch)

        data = []
        if self.policy.module_args.return_rlhf_data:
            data.append(policy_out_queue[3])
        if self.reference.module_args.return_rlhf_data:
            data.append(ref_out_queue[1])
        if self.reward.module_args.return_rlhf_data:
            data.append(reward_out_queue)
        return self.get_all_merged_data(data, out_queue, encode=False)

    def make_experiences(self):
        """
        Generate a collection of experiences for one episode
        """
        data_queue = Queue()
        out_queue = Queue()
        # TODO: generate data_flow by ast parser
        policy_out_queues = [Queue() for i in range(4)]
        ref_out_queue = [Queue() for i in range(2)]
        old_value_out_queue = Queue()
        reward_out_queue = Queue()

        policy_iter = cycle(iter(self.policy.replicas))
        if self.args.colocation:
            for mb in range(self.batch_per_episode):
                # TODO: independent data loader
                policy = next(policy_iter)
                query = policy.master.next_batch.remote()
                data_queue.put(self.encode_data(mb, query))
            self.generate_loop_sync(data_queue, policy_out_queues, ref_out_queue, old_value_out_queue, reward_out_queue,
                                    out_queue, self.batch_per_episode)
        else:
            for mb in range(self.batch_per_episode):
                # TODO: independent data loader
                policy = next(policy_iter)
                query = policy.master.next_batch.remote()
                data_queue.put(self.encode_data(mb, query))
                data = self.generate_step(data_queue, policy_out_queues, ref_out_queue, old_value_out_queue,
                                          reward_out_queue)
                out_queue.put(data)

        for policy in self.policy.replicas:
            ref = policy.master.add_step.remote(self.batch_per_episode)
            future.get(ref)
        return out_queue
