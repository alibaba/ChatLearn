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

import importlib
import math
from itertools import cycle

from chatlearn.data.ranking import batch_generation_ranking
from chatlearn.utils import future
from chatlearn.utils.logger import logger
from .executor import Executor

vllm_exist = importlib.util.find_spec("vllm")
if vllm_exist:
    from chatlearn.models.vllm_module import RLHFVLLMModule


class Environment(Executor):
    """BaseEnv"""

    def __init__(self, models):
        """
        Environment

        Args
        ----
        models : List[RLHFModule]
            a list of modules
        """
        super().__init__(models)
        for model in self.models:
            model.register_func("forward_step")
        self._batch_size = None
        self._batch_per_episode = None
        self._dataset = None
        self.data_iter = None
        self._padding_config = {}
        self.merged_buffer = {}
        self.model2iter = {}

    def set_dataset(self, dataset):
        self._dataset = dataset
        return self

    def setup_dataset(self):
        self.data_producer = self.models[0]
        assert self.sample_per_episode % len(self.data_producer.replicas) == 0, \
            "replica number of data producer model must be divisible by sample_per_episode"
        self.sample_per_episode_per_replica = self.sample_per_episode // len(self.data_producer.replicas)
        logger.info("start set dataset for data_producer")
        refs = []
        if self.models[0].module_args.batch_generation.ranking:
            episode_per_epoch = math.ceil(len(self._dataset) / self.sample_per_episode)
            self._dataset = batch_generation_ranking(self._dataset, episode_per_epoch, self.sample_per_episode)
        for policy_replica in self.data_producer.replicas:
            ref = policy_replica.master._build_dataloader.remote(self._dataset,
                                                                 self.batch_size,
                                                                 self.sample_per_episode_per_replica)
            refs.append(ref)
        future.get(refs)
        logger.info("set dataset for data_producer done")


    def setup(self):
        self.use_vllm_backend = vllm_exist and isinstance(self.models[0].replicas[0].model, RLHFVLLMModule)
        super().setup()
        self.setup_dataset()

        for model_node in self.model_flow.model_nodes:
            model = model_node.model.replicas[0]
            config = future.get(model.master.padding_config.remote())
            self._padding_config.update(config)

        if self.use_vllm_backend:
            # setup vllm scheduler
            refs = []
            for model_replica in self.models[0].replicas:
                ref = model_replica.tailer.build_scheduler.remote()
                refs.append(ref)
            future.get(refs)

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
            num_batch = self.sample_per_episode // (num_replica * self.batch_size) * num_replica
            remainder = self.sample_per_episode % (num_replica * self.batch_size)
            if remainder >= num_replica:
                self._batch_per_episode = num_batch + num_replica
            else:
                self._batch_per_episode = num_batch + remainder
        return self._batch_per_episode

    def vllm_post_process_outputs(self, replica):
        """post precess of results in current episode"""
        return replica.tailer.decode.remote()

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

    def generate_step(self, data_queue, step):
        for i, model_node in enumerate(self.model_flow.model_nodes):
            if i == 0 and self.use_vllm_backend:
                self.vllm_post_process_generate_step_one_model(model_node.model, model_node.out_queues, step)
                continue
            input_queues = data_queue if i == 0 else model_node.get_input_queues()
            self.generate_step_one_model(model_node.model, input_queues, model_node.out_queues, to_empty_cache=False,
                                         step_num=step)
        data = []
        for model_node in self.model_flow.model_nodes:
            if model_node in self.model_flow.return_model_nodes:
                data.append(model_node.out_queues[-1])
        return self.get_merged_data(data, encode=False)

    def add_request(self, is_eval=False):
        request_rets = []
        for model_replica in self.models[0].replicas:
            query = model_replica.master.next_batch.remote(is_eval=is_eval)
            ret = model_replica.tailer._add_request.remote(query)
            request_rets.append(ret)
        future.get(request_rets)

    def make_experiences(self):
        """
        Generate a collection of experiences for one episode
        """
        # Assume the first model produce the data
        # data_producer = self.model_flow.model_nodes[0].model
        data_queues, out_queue = self.setup_queues()

        if self.use_vllm_backend:
            data_queue = data_queues[0]
            for mb in range(self.batch_per_episode):
                # add requests of current episode to vllm scheduler
                self.add_request()

                # eval loop of current episode
                num_remaining_request = True
                while num_remaining_request:
                    step_output_rets = []
                    for model_replica in self.data_producer.replicas:
                        query = model_replica.master.schedule.remote()
                        data_queue.put(self.encode_data(mb, query))
                        data, _ = self.generate_step_one_model_internal(self.model_flow.model_nodes[0].model,
                                                                        data_queue, mb)
                        step_output_rets.append(data)
                    num_remaining_request = future.get(step_output_rets)[0][0]
                data = self.generate_step(None, mb)
                out_queue.put(data)
        else:
            data_producer_iter = cycle(iter(self.data_producer.replicas))
            for mb in range(self.batch_per_episode):
                current_data_producer = next(data_producer_iter)
                query = current_data_producer.master.next_batch.remote()
                encoded_data = self.encode_data(mb, query)
                for data_queue in data_queues:
                    data_queue.put(encoded_data)
            self.compute_loop(data_queues, out_queue, self.batch_per_episode)
        return out_queue
