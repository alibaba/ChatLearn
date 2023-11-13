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
import importlib
from itertools import cycle

from ray.util.queue import Queue

from chatlearn.utils import future
from chatlearn.utils.logger import logger
from chatlearn.data.ranking import batch_generation_ranking

vllm_exist = importlib.util.find_spec("vllm")
if vllm_exist:
    from chatlearn.models.vllm_module import RLHFVLLMModule


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
        self.models = [policy, reference, reward, value]
        self.policy = policy
        self.reference = reference
        self.reward = reward
        self.value = value
        self.remote_models = [policy, reference, reward, value]
        assert self.sample_per_episode % len(self.policy.replicas) == 0, \
            "replica number of policy model must be divisible by sample_per_episode"
        self.sample_per_episode_per_replica = self.sample_per_episode // len(self.policy.replicas)
        self._batch_size = None
        self._batch_per_episode = None
        self._dataset = None
        self.data_iter = None
        self._padding_config = {}
        self.merged_buffer = {}
        self.model2iter = {}
        self.model2group = {}
        self.reference_value_colocate = []
        self.num_reference_value_to_process = 1
        self.reference_value_results = []
        self.use_vllm_backend = vllm_exist and isinstance(self.policy.replicas[0].model, RLHFVLLMModule)

    @property
    def sample_per_episode(self):
        return self.args.sample_per_episode

    @property
    def batch_size(self):
        if self._batch_size is not None:
            return self._batch_size
        if self.use_vllm_backend:
            num_replica = len(self.models[0].replicas)
            self._batch_size = self.sample_per_episode // num_replica
            if self.models[0].module_args.args_dict.get("vllm_micro_batch_size") is not None and \
                    self.models[0].module_args.args_dict["vllm_micro_batch_size"] != -1:
                self._batch_size = self.models[0].module_args.args_dict["vllm_micro_batch_size"]
        else:
            self._batch_size = self.models[0].module_args.generation_batch_size

        return self._batch_size

    @property
    def batch_per_episode(self):
        if self._batch_per_episode is not None:
            return self._batch_per_episode
        num_replica = len(self.models[0].replicas)
        if self.use_vllm_backend:
            self._batch_per_episode = math.ceil(self.sample_per_episode_per_replica / self.batch_size)
        else:
            num_batch = self.sample_per_episode // (num_replica*self.batch_size) * num_replica
            remainder = self.sample_per_episode % (num_replica*self.batch_size)
            if remainder >= num_replica:
                self._batch_per_episode = num_batch + num_replica
            else:
                self._batch_per_episode = num_batch + remainder
        return self._batch_per_episode

    def setup(self, model_packs=None):
        assert isinstance(model_packs, list), \
            f"model_packs for PPOEnv must be a list, but got {type(model_packs)}"
        refs = []
        for policy_replica in self.policy.replicas:
            ref = policy_replica.master._build_dataloader.remote(self._dataset,
                                                                 self.batch_size,
                                                                 self.sample_per_episode_per_replica)
            refs.append(ref)
        future.get(refs)
        logger.info("set dataset for policy")

        if self.use_vllm_backend:
            # set up scheduler and add request
            refs = []
            for policy_replica in self.policy.replicas:
                ref = policy_replica.master.build_scheduler.remote()
                refs.append(ref)
            future.get(refs)

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

    def set_dataset(self, dataset):
        if self.models[0].module_args.batch_generation.ranking:
            episode_per_epoch = math.ceil(len(dataset) / self.sample_per_episode)
            self._dataset = batch_generation_ranking(dataset, episode_per_epoch, self.sample_per_episode)
        else:
            self._dataset = dataset

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

    def vllm_post_process_outputs(self, replica):
        """post precess of results in current episode"""
        return replica.master.decode.remote()

    def vllm_post_process_generate_step_one_model(self, model, out_queue, mb):
        """
        Args:
            model: DistModel
            out_queue: Queue
        """
        replica = self._get_model(model)
        output = self.vllm_post_process_outputs(replica)

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

    def generate_step_one_model_internal(self, model, in_queue, step_num, func_name="forward_step", to_empty_cache=None, is_eval=False):
        """
        Args:
            model: DistModel
            in_queue: Queue
            step_num: int
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
        kwargs = {}
        if not isinstance(query, list):
            query = [query]
        #if isinstance(query, list):
        replica_num = len(model.replicas)
        last_step_start = max(self.batch_per_episode - replica_num, 0)
        is_last_batch = step_num >= last_step_start
        kwargs["is_last_batch"] = is_last_batch
        if to_empty_cache is not None:
            kwargs["to_empty_cache"] = to_empty_cache
        if is_eval is not None:
            kwargs["is_eval"] = is_eval
        output = func(*query, **kwargs)
        return output, mb

    def has_unfinished_requests(self):
        rets = []
        for model_replica in self.models[0].replicas:
            rets.append(model_replica.master.has_unfinished_requests.remote())
        return all(future.get(rets))

    def generate_step_one_model(self, model, in_queue, out_queue, step_num, func_name="forward_step", to_empty_cache=None, is_eval=False):
        """
        Args:
            model: DistModel
            in_queue: Queue
            out_queue: Queue
            step_num: int
            func_name: str
            to_empty_cache: None or boolean
        """
        output, mb = self.generate_step_one_model_internal(model, in_queue, step_num, func_name, to_empty_cache, is_eval)

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
            to_empty_cache = step >= last_step_start
            _, data = self.generate_step_one_model(model, in_queue, out_queue, step, func_name, to_empty_cache)
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


    def generate_step(self, data_queue, policy_out_queue, ref_out_queue, old_value_out_queue, reward_out_queue, step, mb=0):
        # TODO: generate data_flow by ast parser
        if self.use_vllm_backend:
            self.vllm_post_process_generate_step_one_model(self.policy, policy_out_queue, mb)
        else:
            self.generate_step_one_model(self.policy, data_queue, policy_out_queue, to_empty_cache=False, step_num=step)
        self.generate_step_one_model(self.reference, policy_out_queue[0], ref_out_queue, to_empty_cache=False, step_num=step)
        self.generate_step_one_model(self.value, policy_out_queue[1], old_value_out_queue, to_empty_cache=False, step_num=step)
        self.generate_step_one_model(self.reward, [policy_out_queue[2], ref_out_queue[0], old_value_out_queue],
                                     reward_out_queue, to_empty_cache=False, step_num=step)
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

    def add_request(self, is_eval=False):
        request_rets = []
        for model_replica in self.models[0].replicas:
            query = model_replica.master.next_batch.remote(is_eval=is_eval)
            ret = model_replica.master._add_request.remote(query)
            request_rets.append(ret)
        future.get(request_rets)

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

        if self.use_vllm_backend:
            for mb in range(self.batch_per_episode):
                # add requests of current episode to vllm scheduler
                self.add_request()

                # eval loop of current episode
                while self.has_unfinished_requests():
                    step_output_rets = []
                    for model_replica in self.policy.replicas:
                        query = model_replica.master.schedule.remote()
                        data_queue.put(self.encode_data(mb, query))
                        data, _ = self.generate_step_one_model_internal(self.policy, data_queue, mb)
                        step_output_rets.append(data)
                    future.get(step_output_rets)
                data = self.generate_step(None, policy_out_queues, ref_out_queue, old_value_out_queue,
                                          reward_out_queue, mb)
                out_queue.put(data)
        else:
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
                                            reward_out_queue, mb)
                    out_queue.put(data)
        return out_queue
