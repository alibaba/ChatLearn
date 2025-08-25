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
"""rule reward"""
from typing import Dict, List, Any
from collections import deque, defaultdict
import uuid
import random
import time
import numpy as np

import torch
from transformers import AutoTokenizer

from chatlearn.data.prompt_dataset import PromptPipeline
from chatlearn.runtime.decorator import timeit, compute_decorator, monitor_error
from chatlearn import BaseModule

def max_gen_length_each_round(seq_len:int, num_round: int) -> List[int]:
    total_length = sum(x**2 for x in range(seq_len))
    target = total_length // num_round
    current_sum = 0
    slice_idx = [0]
    for i in range(seq_len):
        current_sum += i ** 2
        if current_sum > target:
            slice_idx.append(i)
            current_sum = 0
    slice_idx.append(seq_len)
    return [a - b for a, b in zip(slice_idx[1:], slice_idx[:-1])]

class RolloutManager(BaseModule):
    """rule reward"""

    @timeit("rollout_manager_setup")
    @monitor_error("rollout_manager_setup")
    def setup(self):
        self._metric_prefix = "rollout_manager"
        self.rollout_finished_no_train = defaultdict(list)
        self.num_response_track = defaultdict(int)
        self.rollout_not_finished = []
        self.max_rollout_round = self.module_args.get("max_rollout_round")
        self.max_gen_len = self.module_args.get("max_gen_len")
        self.max_token_per_round = max_gen_length_each_round(self.max_gen_len, self.max_rollout_round)
        print(f"debugyy token_per_round: {self.max_token_per_round}")
        self.num_inference_per_prompt = self.module_args.get("num_inference_per_prompt")
        self.mini_response_per_prompt = self.module_args.get("mini_response_per_prompt")
        self.mertic_dict = {}

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

    def initialze_data(self, data: List[Dict[str, Any]], first_stage: bool = True):
        # Add extra key for partial rollout
        # prompt_uid: prompt_uid for tracking completion for prompts
        # rollout_round: tracking rollout round for partial rollout
        # str_outputs: buffer for all str outputs
        # prompt_token_length: prompt token length before rollout
        # prompt_token_ids: prompt id before rollout
        # max_generate_token_length: max token can be generated with single round
        for sample in data:
            sample["uuid"] = uuid.uuid4()
            sample["prompt_uid"] = hash(sample["prompt"])
            sample["rollout_round"] = 0
            sample["max_generate_token_length"] = self.max_token_per_round[sample["rollout_round"]]
        return data

    @timeit("get_sample_for_rollout")
    @compute_decorator(trainable=False, rollout=False)
    def get_sample_for_rollout(self, data: List[Dict[str, Any]], **kwargs):
        # Get sample_per_episode samples from prompts_dataset
        # Add these samples into self.rollout_not_finished for future rollout
        # Get first sample_per_episode samples from self.rollout_not_finished for this round rollout
        sample_per_episode = len(data)
        data = self.initialze_data(data)
        self.rollout_not_finished.extend(data)
        train_batch = self.rollout_not_finished[:sample_per_episode]
        # Record start episode id
        for single_data in train_batch:
            if "start_episode" not in single_data:
                single_data["start_episode"] = self._episode_id
        random.shuffle(train_batch)
        round_track = {f"round_{i}_samples": sum(d.get("rollout_round") == i for d in train_batch) for i in range(self.max_rollout_round)}
        self.mertic_dict.update(round_track)
        return train_batch

    def is_finished(self, data_b):
        # determine whether the rollout is finished
        #print(f"response len: {data_b["response_token_length"]}, rollout round: {data_b["rollout_round"]}")
        return (data_b["response_token_length"] < data_b["max_generate_token_length"]) or \
            (data_b["rollout_round"] == self.max_rollout_round)

    def find_index_by_uuid(self, sample_per_episode, uuid):
        idx = next(i for i,d in enumerate(self.rollout_not_finished[:sample_per_episode]) if d['uuid'] == uuid)
        return idx

    def update_data(self, data, rollout_result, is_finished):
        # Update data in self.rollout_not_finished buffer for single rollout round
        # - Append str_outputs
        # - Update rollout_round
        # - Add response_token_length
        # - Update input_ids for next rollout round
        # - Update all_tokens

        assert data["uuid"] == rollout_result["uuid"]
        data.update({
            "uuid": rollout_result["uuid"],
            "str_outputs": data.get("str_outputs", "") + rollout_result["str_outputs"],
            "rollout_round": rollout_result["rollout_round"],
            "response_token_length": data.get("response_token_length", 0) + rollout_result["response_token_length"],
            "input_ids": rollout_result["all_tokens"].tolist(),
            "all_tokens": rollout_result["all_tokens"],
            "max_generate_token_length": self.max_token_per_round[min(rollout_result["rollout_round"], len(self.max_token_per_round) - 1)]
        })
        return data
    
    def logging_generate_by_round(self, rollout_result_list):
        logging_generate = {f"round_{i}_response": [] for i in range(self.max_rollout_round)}
        for data in rollout_result_list:
            logging_generate[f"round_{data["rollout_round"] - 1}_response"].append(data["response_token_length"])
        update_dict = {}
        for key in logging_generate:
            update_dict[f"{key}_mean"] = 0 if len(logging_generate[key]) == 0 else np.mean(np.array(logging_generate[key]))
            update_dict[f"{key}_std"] = 0 if len(logging_generate[key]) == 0 else np.std(np.array(logging_generate[key]))
        self.mertic_dict.update(update_dict)
        self._metric_list.append(self.mertic_dict)

    @timeit("post_process_rollout_results")
    @compute_decorator(trainable=False, rollout=False)
    def post_process_rollout_results(self, rollout_result_list, **kwargs):
        self.logging_generate_by_round(rollout_result_list)
        start = time.time()
        sample_per_episode = len(rollout_result_list)
        finished_uuid = []
        unfinished_data = []
        for sample in rollout_result_list:
            uuid = sample["uuid"]
            prompt_uid = sample["prompt_uid"]
            finished = self.is_finished(sample)
            data_idx = self.find_index_by_uuid(sample_per_episode, uuid)
            data_b = self.update_data(self.rollout_not_finished[data_idx], sample, finished)
            if finished:
                # Finished, add data to self.rollout_finished_no_train[prompt_uid]
                self.rollout_finished_no_train[prompt_uid].append(data_b)
                self.num_response_track[prompt_uid] += 1
                finished_uuid.append(uuid)
            else:
                # If not finished, update data in rollout_not_finished
                unfinished_data.append(data_b)
        # update remaining data
        unfinished_data.extend(self.rollout_not_finished[sample_per_episode:])
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
        print(f"debugyy final sum train: {len(train_data)}")
        print("data preprocess time: %.3f" % (time.time() - start), flush=True)
        return train_data