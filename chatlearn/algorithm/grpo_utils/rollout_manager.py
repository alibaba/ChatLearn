# Copyright 2024 Alibaba-inc. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
"""Rollout Manager"""
import uuid
import random
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

from chatlearn.data.prompt_dataset import PromptPipeline
from chatlearn.runtime.decorator import timeit, compute_decorator, monitor_error
from chatlearn import BaseModule

class RolloutManager(BaseModule):
    """Rollout Manager"""
    def setup(self):
        self._metric_prefix = "rollout_manager"
        self.rollout_finished_no_train = defaultdict(list)
        self.num_response_track = defaultdict(int)
        self.rollout_not_finished = []
        self.max_rollout_round = self.module_args.get("max_rollout_round")
        self.max_gen_len = self.module_args.get("max_gen_len")
        self.ratio = self.module_args.get("rollout_ratio")
        self.max_token_per_round = [int(self.max_gen_len * ratio) for ratio in self.ratio]
        self.num_inference_per_prompt = self.module_args.get("num_inference_per_prompt")
        self.mini_response_per_prompt = self.module_args.get("mini_response_per_prompt")
        self.metric_dict = {}

    def build_dataset(self, prompts: List[Dict], is_eval=False):
        # prompts seems like the total data set by engine.set_dataset(dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(self.module_args['load'], trust_remote_code=True)
        prompts_dataset = PromptPipeline(
            prompts,
            sum(self.max_token_per_round),
            self.tokenizer,
            enable_thinking=self.module_args.get("enable_thinking", False),
        )
        return prompts_dataset

    def initialze_data(self, data: List[Dict[str, Any]]):
        for sample in data:
            sample["uuid"] = uuid.uuid4()
            sample["prompt_uid"] = hash(sample["prompt"])
            sample["rollout_round"] = 0
            sample["max_generate_token_length"] = self.max_token_per_round[sample["rollout_round"]]
        return data

    @monitor_error()
    @compute_decorator(trainable=False, rollout=False)
    @timeit()
    def get_sample_for_rollout(self, data: List[Dict[str, Any]], **kwargs): # pylint: disable=unused-argument
        # Get sample_per_episode samples from prompts_dataset
        # Add these samples into self.rollout_not_finished for future rollout
        # Send all samples to rollout engine
        data = self.initialze_data(data)
        self.rollout_not_finished.extend(data)
        train_batch = self.rollout_not_finished
        # Record start episode id
        for single_data in train_batch:
            if "start_episode" not in single_data:
                single_data["start_episode"] = self._episode_id
        random.shuffle(train_batch)
        round_track = {f"round_{i}_samples": sum(d.get("rollout_round") == i for d in train_batch) for i in range(self.max_rollout_round)}
        self.metric_dict.update(round_track)
        return train_batch

    def is_finished(self, data_b: Dict[str, Any]):
        # determine whether the rollout is finished
        # rollout finished if
        # 1. response_token_lenght is less than this round's max_generate_token_length
        # 2. reach max rollout round

        return (data_b["response_token_length"] < data_b["max_generate_token_length"]) or \
            (data_b["rollout_round"] == self.max_rollout_round)

    def find_index_by_uuid(self, _uuid):
        idx = next(i for i,d in enumerate(self.rollout_not_finished) if d['uuid'] == _uuid)
        return idx

    def update_data(self, data: Dict[str, Any], rollout_result: Dict[str, Any]):
        # Merge data in self.rollout_not_finished buffer and rollout_result with same uuid
        assert data["uuid"] == rollout_result["uuid"]
        data.update({
            "uuid": rollout_result["uuid"],
            "str_outputs": data.get("str_outputs", "") + rollout_result["str_outputs"],
            "rollout_round": rollout_result["rollout_round"],
            "response_token_length": data.get("response_token_length", 0) + rollout_result["response_token_length"],
            "input_ids": rollout_result["all_tokens"].tolist(),
            "all_tokens": rollout_result["all_tokens"],
            "max_generate_token_length": self.max_token_per_round[rollout_result["rollout_round"]] \
                if rollout_result["rollout_round"] < self.max_rollout_round else 0
        })
        return data

    def logging_generate_by_round(self, rollout_result_list: List[Dict[str, Any]]):
        # Logging generate metrics
        logging_generate = {f"round_{i}_response": [] for i in range(self.max_rollout_round)}
        for data in rollout_result_list:
            logging_generate[f"round_{data['rollout_round'] - 1}_response"].append(data["response_token_length"])
        update_dict = {}
        for key in logging_generate:
            update_dict[f"{key}_mean"] = 0 if len(logging_generate[key]) == 0 else np.mean(np.array(logging_generate[key]))
            update_dict[f"{key}_max"] = 0 if len(logging_generate[key]) == 0 else np.max(np.array(logging_generate[key]))
            update_dict[f"{key}_min"] = 0 if len(logging_generate[key]) == 0 else np.min(np.array(logging_generate[key]))
        self.metric_dict.update(update_dict)

    @monitor_error()
    @compute_decorator(trainable=False, rollout=False)
    @timeit()
    def post_process_rollout_results(self, rollout_result_list: List[Dict[str, Any]], **kwargs): # pylint: disable=unused-argument
        self.logging_generate_by_round(rollout_result_list)
        finished_uuid = []
        unfinished_data = []
        for sample in rollout_result_list:
            _uuid = sample["uuid"]
            prompt_uid = sample["prompt_uid"]
            finished = self.is_finished(sample)
            data_idx = self.find_index_by_uuid(_uuid)
            # Merge data from buffer and data from rollout
            data_b = self.update_data(self.rollout_not_finished[data_idx], sample)
            if finished:
                # Finished, add data to self.rollout_finished_no_train[prompt_uid]
                self.rollout_finished_no_train[prompt_uid].append(data_b)
                self.num_response_track[prompt_uid] += 1
                finished_uuid.append(_uuid)
            else:
                # If not finished, update data in rollout_not_finished
                unfinished_data.append(data_b)
        # update remaining data
        self.rollout_not_finished = unfinished_data
        train_data = []
        pop_keys = []
        for key, data_list in self.rollout_finished_no_train.items():
            if self.num_response_track[key] > self.mini_response_per_prompt:
                train_data.extend(data_list)
                pop_keys.append(key)
        for key in pop_keys:
            self.rollout_finished_no_train.pop(key)
            if self.num_response_track[key] == self.num_inference_per_prompt:
                self.num_response_track.pop(key)
        random.shuffle(train_data)
        total_train_token = sum(d['response_token_length'] + d['prompt_token_length'] for d in train_data)
        self.metric_dict.update({'total_valid_tokens': total_train_token, 'num_train_samples': len(train_data)})
        self._metric_list.append(self.metric_dict)
        return train_data
