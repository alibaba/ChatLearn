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
"""entry file"""

import random

import numpy
import torch
from models import PolicyInference
from models import PolicyReference
from models import PolicyTrainer
from models import RewardInference
from models import ValueInference
from models import ValueTrainer
from models.utils import write_jsonl, read_jsonl, tensorboard_scalar_dict, listdict_to_dictlist
from torch.utils.tensorboard import SummaryWriter

import chatlearn
from chatlearn import Evaluator
from chatlearn import RLHFEngine


def get_prompts(fp, num_limit=-1):
    prompts_jsons = read_jsonl(fp)

    if "text" in prompts_jsons[0]:
        prompts = [p["text"] for p in prompts_jsons]
        patten = '\n\nAssistant: '
        prompts = [prompt[:prompt.find(patten) + len(patten)] for prompt in prompts]
        if num_limit != -1:
            prompts = prompts[:num_limit]
        return prompts
    elif 'prompt' in prompts_jsons[0]:
        prompts = [p["prompt"] for p in prompts_jsons]
        if num_limit != -1:
            prompts = prompts[:num_limit]
        return prompts
    else:
        prompts = [p["query"] for p in prompts_jsons]
        if num_limit != -1:
            prompts = prompts[:num_limit]
        formatted_prompts = [f"\n\nHuman: {p}\n\nAssistant: " for p in prompts]
        return formatted_prompts


if __name__ == "__main__":
    chatlearn.init()
    args = chatlearn.get_args()
    policy_model = PolicyInference("policy")
    value_model = ValueInference("value")
    reference_model = PolicyReference("reference")
    reward_model = RewardInference("reward")
    ppo_policy_model = PolicyTrainer("ppo_policy")
    ppo_value_model = ValueTrainer("ppo_value")
    if args.rlhf_args.eval_episode_interval > 0:
        policy_model.register_eval_func("eval_forward")
        reward_model.register_eval_func("eval_forward")
    engine = RLHFEngine(policy_model, reference_model, reward_model, value_model, ppo_policy_model, ppo_value_model)
    all_prompts = get_prompts(args.rlhf_args.data_path, num_limit=args.rlhf_args._args_dict['training_data_num_limit'])
    random.seed(policy_model.model_args["seed"])
    split_ratio = 0.9 if args.rlhf_args.eval_episode_interval > 0 else 1
    num_train = int(len(all_prompts) * split_ratio)
    random.shuffle(all_prompts)
    train_prompts = all_prompts[:num_train]
    policy_checkpoint = policy_model.model_args.get("load", 0)
    exp_name = policy_model.model_args["exp_name"]


    def eval_post_process(results, eval_info):
        results = listdict_to_dictlist(results)
        writer = SummaryWriter(
            log_dir=args.models["policy"].args_dict['tensorboard_dir'],
            max_queue=99999)

        eval_reward_stats = {"eval_reward_mean": numpy.mean(results['rewards'])}
        train_iteration = eval_info["train_iteration"]

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                torch.distributed.get_world_size() - 1):
                tensorboard_scalar_dict(writer, prefix="eval_reward_each/",
                                        global_step=train_iteration,
                                        scalar_dict=eval_reward_stats)

        else:
            tensorboard_scalar_dict(writer, prefix="eval_reward_each/",
                                    global_step=train_iteration,
                                    scalar_dict=eval_reward_stats)

        save_fp = f"{args.rlhf_args._args_dict['eval_output_dir']}/{exp_name}/{train_iteration}/eval_json_res.json"
        write_jsonl(results["eval_jsonl"], save_fp)


    if args.rlhf_args.eval_episode_interval > 0:
        val_prompts = all_prompts[num_train:]
        eval_num_limit = args.rlhf_args.get('eval_data_num_limit')
        if eval_num_limit:
            eval_num_limit = min(eval_num_limit, len(val_prompts))
            val_prompts = val_prompts[:eval_num_limit]
        evaluator = Evaluator([policy_model, reward_model]) \
            .set_dataset(val_prompts) \
            .set_post_process_func(eval_post_process)
        engine.set_evaluator(evaluator)
    engine.set_dataset(train_prompts)
    engine.learn()
