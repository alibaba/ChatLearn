"""examples"""
import argparse

import numpy
import torch
import yaml
from datasets import load_dataset
from pathlib import Path
from torch.utils.data import Dataset

import rlhf
from models.old_policy_inference import PolicyMegatronInference as PolicyModel
from models.old_value_inference import ValueMegatronInference as ValueModel
from models.policy_trainer import MegatronPolicy
from models.reference import PolicyReference as ReferenceModel
from models.reward_inference import RewardModelMegatronInference as RewardModel
from models.value_trainer import ValueMegatronTrainer
from rlhf import Evaluator
from rlhf import RLHFEngine
from utils.utils import write_jsonl, read_jsonl, tensorboard_scalar_dict, listdict_to_dictlist


class RLHFDataset(Dataset):
    def __init__(self, train_path, tokenizer, max_length):
        self.post_list = []
        dataset = load_dataset(train_path, split='train')
        num_samples = 2000
        for i, sample in enumerate(dataset):
            if i == num_samples:
                break
            self.post_list.append(sample["prompt"] + sample["label"])

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        txt = self.post_list[idx]
        return {"prompts": txt[:self.max_length]}


def get_prompts(fp, num_limit=-1):
    prompts_jsons = read_jsonl(fp)
    prompts = [p["query"] for p in prompts_jsons]
    if num_limit != -1:
        prompts = prompts[:num_limit]
    return prompts


def get_prompts_from_sft(fp, policy_model_args, num_limit=-1):
    prompts_jsons = read_jsonl(fp)

    if "text" in prompts_jsons[0]:
        prompts = [p["text"] for p in prompts_jsons]
        patten = '\n\nAssistant: '
        prompts = [patten.join(line.split(patten)[:-1]) for line in prompts]
        if num_limit != -1:
            prompts = prompts[:num_limit]
        return prompts
    elif 'prompt' in prompts_jsons[0]:
        prompts = [p["prompt"] for p in prompts_jsons]
        if num_limit != -1:
            prompts = prompts[:num_limit]
        print(f"Get prompt size {len(prompts)}", flush=True)
        return prompts
    else:
        prompts = [p["query"] for p in prompts_jsons]
        # prompts = list(set(prompts))
        if num_limit != -1:
            prompts = prompts[:num_limit]

        formatted_prompts = [f"\n\nHuman: {p}\n\nAssistant: " for p in prompts]
        return formatted_prompts


def parse_args_from_yaml(config_file):
    with open(config_file, 'r', encoding='utf-8') as stream:
        config_vars = yaml.load(stream, Loader=yaml.FullLoader)

        return config_vars


def get_config_dir():
    parser = argparse.ArgumentParser(description='RLHF Arguments',
                                     allow_abbrev=False)

    parser.add_argument("-c", "--config",
                        required=True,
                        help="where to load YAML configuration",
                        metavar="FILE")

    args = parser.parse_args()
    return Path(args.config).parent.as_posix()


def copy_configs_to_dir(src_dir, dst_dir):
    import shutil
    if Path(dst_dir).exists():
        shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir)


def copy_configs_to_configs_exp_name_dir():
    config_dir = get_config_dir()
    base_config = parse_args_from_yaml(f"{config_dir}/base.yaml")
    exp_name = base_config["exp_name"]
    log_dir = base_config["log_dir"]
    copy_configs_to_dir(config_dir, dst_dir=f"{log_dir}/{exp_name}/")


if __name__ == "__main__":

    # copy_configs_to_configs_exp_name_dir()

    rlhf.init()
    args = rlhf.get_args()

    models = {}
    policy_model = PolicyModel("policy")
    value_model = ValueModel("value")
    reference_model = ReferenceModel("reference")
    reward_model = RewardModel("reward")
    ppo_policy_model = MegatronPolicy("ppo_policy")
    ppo_value_model = ValueMegatronTrainer("ppo_value")

    engine = RLHFEngine(policy_model, reference_model, reward_model, value_model, ppo_policy_model, ppo_value_model)
    # all_prompts = get_prompts_from_sft(args.rlhf_args.data_path, num_limit=args.rlhf_args._args_dict['training_data_num_limit'])

    all_prompts = get_prompts_from_sft(args.rlhf_args.data_path, policy_model.model_args,
                                       num_limit=args.rlhf_args._args_dict['training_data_num_limit'], )
    print(f"%%%%%%%%%%%%%%%%%%%%%rlhf_args: {args.rlhf_args}")
    print(f"%%%%%%%%%%%%%%%%%%%%%args: {args}")

    len_all_prompts = len(all_prompts)
    # train_len = len_all_prompts * 2 // 3
    # val_len = len_all_prompts - train_len
    # train_prompts = all_prompts[:train_len]
    # val_prompts = [" 赵丽颖演过哪些历史剧？"]* 1000
    val_prompts = get_prompts_from_sft(args.rlhf_args._args_dict["eval_data_path"], policy_model.model_args,
                                       num_limit=args.rlhf_args._args_dict['eval_data_num_limit'], )

    import random

    random.seed(policy_model.model_args["seed"])
    train_prompts = random.sample(all_prompts, len_all_prompts * 9 // 10)
    print(f"sss: {policy_model.model_args}")
    policy_checkpoint = policy_model.model_args.get("load", 0)
    exp_name = policy_model.model_args["exp_name"]


    def eval_post_process(results, eval_info):

        results = listdict_to_dictlist(results)

        from tensorboardX import SummaryWriter
        writer = SummaryWriter(
            log_dir=args.models["policy"].args_dict['tensorboard_dir'],
            max_queue=99999)

        eval_reward_stats = {"eval_reward_mean": numpy.mean(results['rewards'])}
        train_iteration = eval_info["train_iteration"]

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                torch.distributed.get_world_size() - 1):
                # actual log
                print(f"%%%%%%%%%%" * 10)
                print(f"eval reward stats: {eval_reward_stats} iter: {train_iteration}")
                tensorboard_scalar_dict(writer, prefix=f"eval_reward_each/",
                                        global_step=train_iteration,
                                        scalar_dict=eval_reward_stats)

        else:
            # actual log
            print(f"%%%%%%%%%%" * 10)
            print(f"eval reward stats: {eval_reward_stats} iter: {train_iteration}")
            tensorboard_scalar_dict(writer, prefix=f"eval_reward_each/",
                                    global_step=train_iteration,
                                    scalar_dict=eval_reward_stats)

        save_fp = f"{args.rlhf_args._args_dict['eval_output_dir']}/{exp_name}/{train_iteration}/eval_json_res.json"

        print(f"eval inference finished: got jsons number: {len(results['eval_jsonl'])}")
        write_jsonl(results["eval_jsonl"], save_fp)


    if args.rlhf_args.eval_episode_interval > 0:
        evaluator = Evaluator([policy_model, reward_model]) \
            .set_dataset(val_prompts) \
            .set_post_process_func(eval_post_process) \
            .register_func("policy", "eval_forward") \
            .register_func("reward", "eval_forward")

        engine.set_evaluator(evaluator)

    engine.set_dataset(train_prompts)
    engine.learn()
