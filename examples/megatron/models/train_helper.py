# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
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
"""training helper"""

import numpy
import torch

from megatron.training.global_vars import get_tensorboard_writer, get_wandb_writer

import chatlearn
from .utils import write_jsonl, read_jsonl, tensorboard_scalar_dict, wandb_scalar_dict, listdict_to_dictlist

def eval_post_process(results, eval_info):
    """
    Post process function for model evaluation results.

    Args
        ----
        post_process_func

            This function accept two arguments.
            1. results: a list of evaluation results
            2. eval_info: a dict meta that contains "train_iteration" and "episode_iteration"
    """
    args = chatlearn.get_args().runtime_args
    results = results["reward"]
    results = listdict_to_dictlist(results)
    if args.get('eval_data_num_limit') > 0:
        assert len(results['rewards']) == args.get('eval_data_num_limit'), f"expect {len(results['rewards'])} == {args.get('eval_data_num_limit')}"
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()

    eval_reward_stats = {"eval_reward_mean": numpy.mean(results['rewards'])}
    train_iteration = eval_info["train_iteration"]

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == (
            torch.distributed.get_world_size() - 1):
            tensorboard_scalar_dict(writer, prefix="eval_reward_each/",
                                    global_step=train_iteration,
                                    scalar_dict=eval_reward_stats)
            if wandb_writer:
                wandb_scalar_dict(wandb_writer, prefix="eval_reward_each/",
                                        global_step=train_iteration,
                                        scalar_dict=eval_reward_stats)
    else:
        tensorboard_scalar_dict(writer, prefix="eval_reward_each/",
                                global_step=train_iteration,
                                scalar_dict=eval_reward_stats)
        if wandb_writer:
            wandb_scalar_dict(wandb_writer, prefix="eval_reward_each/",
                                    global_step=train_iteration,
                                    scalar_dict=eval_reward_stats)

    print(f"eval reward stats: {eval_reward_stats} iter: {train_iteration}")
    save_fp = f"{args.output_dir}/eval/{train_iteration}/eval_json_res.json" # pylint: disable=line-too-long
    write_jsonl(results["eval_jsonl"], save_fp)


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
        prompts = []
        for p in prompts_jsons:
            if "response" in p:
                prompts.append((p["prompt"], p["response"], p["rejected"]))
            elif 'eval_func' in p:
                # math
                prompts.append(p)
            else:
                prompts.append(p["prompt"])
        if num_limit != -1:
            prompts = prompts[:num_limit]
        return prompts
    else:
        prompts = [p["query"] for p in prompts_jsons]
        if num_limit != -1:
            prompts = prompts[:num_limit]
        formatted_prompts = [f"\n\nHuman: {p}\n\nAssistant: " for p in prompts]
        return formatted_prompts
