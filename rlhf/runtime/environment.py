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

from rlhf.utils import future
from rlhf.utils import utils
from rlhf.utils.logger import logger


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
        self.policy = policy
        self.reference = reference
        self.reward = reward
        self.value = value
        self.remote_models = [policy, reference, reward, value]
        self.batch_size = self.policy.module_args.generation_batch_size
        assert self.sample_per_episode % self.batch_size == 0, \
            f"currently sample_per_episode {self.sample_per_episode} should be times of generation_batch_size {self.batch_size}"
        self.batch_per_episode = math.ceil(self.sample_per_episode / self.batch_size)
        self._dataset = None
        self.data_iter = None
        self._padding_config = {}
        self.merged_buffer = {}
        self.model2iter = {}
        self.model2group = {}

    def setup(self):
        data_loader = self.build_data_loader() # pylint: disable=assignment-from-no-return
        if data_loader is not None:
            self.data_iter = iter(data_loader)
        else:
            refs = []
            for i, policy_replica in enumerate(self.policy.replicas):
                ref = policy_replica.master._build_dataloader.remote(self._dataset[i])
                refs.append(ref)
            future.get(refs)
            logger.info("set dataset for policy")

        for dist_model in [self.policy, self.reference, self.reward, self.value]:
            model = dist_model.replicas[0]
            config = future.get(model.master.padding_config.remote())
            self._padding_config.update(config)
        model_names = [m.name for m in self.remote_models]
        # self.args.colocation
        for group in self.args.colocation:
            new_group = []
            for model in group:
                if model in model_names:
                    new_group.append(model)
                    self.model2group[model] = new_group

    def set_dataset(self, dataset, drop_last=False):
        self._dataset = []
        # TODO: compare with use only master dataloader
        data_len = len(dataset)
        data_part_num = self.policy.num_replica
        indices = utils.split_index(data_len, data_part_num)

        for start, end in indices:
            data_part = dataset[start:end]
            drop_len = len(data_part) % self.batch_size
            if drop_len:
                if drop_last:
                    data_part = data_part[:-drop_len]
                else:
                    wrap_len = self.batch_size - drop_len
                    data_part = data_part + data_part[:wrap_len]
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
            # if model has multiple actors, just use the first one
            # TODO: this can be optimized, since only the first return is needed
            # TODO: optimize in forward_step/train_step
            data0 = data0[0]
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
                    data = data[0]
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

    def generate_step_one_model(self, model, in_queue, out_queue, func_name="forward_step"):
        """
        Args:
            model: DistModel
            in_queue: Queue
            out_queue: Queue
        """
        replica = self._get_model(model)

        if isinstance(in_queue, list):
            data = self.get_merged_data(in_queue)
        else:
            data = in_queue.get()
        mb, query = self.decode_data(data)
        func = getattr(replica, func_name)
        if isinstance(query, list):
            output = func(*query)
        else:
            output = func(query)
        if isinstance(output, list):
            output = output[0]
        if isinstance(out_queue, list):
            for oq in out_queue:
                oq.put(self.encode_data(mb, output))
        else:
            out_queue.put(self.encode_data(mb, output))
        return out_queue, output

    def wait_and_empty_cache(self, model, results, func_name):
        future.wait(results, f"{model.name} {func_name}")
        # empty cache, so that other models can use
        refs = model.empty_cache()
        future.get(refs)

    def generate_loop_one_model(self, model, in_queue, out_queue, func_name, to_clear_cache):
        results = []
        for _ in range(self.batch_per_episode):
            _, data = self.generate_step_one_model(model, in_queue, out_queue, func_name)
            results.append(data)
        if model.name in self.model2group and len(self.model2group[model.name]) > 1:
            self.wait_and_empty_cache(model, results, func_name)
        else:
            to_clear_cache.append(model)
            return results

    def generate_step(self, data_queue, policy_out_queue, ref_out_queue, old_value_out_queue, reward_out_queue):
        # TODO: generate data_flow by ast parser
        self.generate_step_one_model(self.policy, data_queue, policy_out_queue)
        self.generate_step_one_model(self.reference, policy_out_queue[0], ref_out_queue)
        self.generate_step_one_model(self.value, policy_out_queue[1], old_value_out_queue)
        self.generate_step_one_model(self.reward, [policy_out_queue[2], ref_out_queue[0], old_value_out_queue],
                                     reward_out_queue)
        data = []
        if self.policy.module_args.return_rlhf_data:
            data.append(policy_out_queue[3])
        if self.reference.module_args.return_rlhf_data:
            data.append(ref_out_queue[1])
        if self.reward.module_args.return_rlhf_data:
            data.append(reward_out_queue)
        return self.get_merged_data(data, encode=False)

    def generate_loop_sync(self, data_queue, policy_out_queue, ref_out_queue, old_value_out_queue, reward_out_queue,
                           out_queue):
        # TODO: generate data_flow by ast parser
        func_name = "forward_step"
        to_clear_cache = []
        self.generate_loop_one_model(self.policy, data_queue, policy_out_queue, func_name, to_clear_cache)
        self.generate_loop_one_model(self.reference, policy_out_queue[0], ref_out_queue, func_name, to_clear_cache)
        self.generate_loop_one_model(self.value, policy_out_queue[1], old_value_out_queue, func_name, to_clear_cache)
        self.generate_loop_one_model(self.reward, [policy_out_queue[2], ref_out_queue[0], old_value_out_queue],
                                     reward_out_queue, func_name, to_clear_cache)

        refs = []
        for model in to_clear_cache:
            refs += model.empty_cache()
        future.wait(refs)

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
                                    out_queue)
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
