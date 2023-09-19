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
"""reward inference"""

import codecs
import json
import re
from collections import defaultdict
from pathlib import Path
from time import time

import torch
from megatron import get_args
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.global_vars import get_tensorboard_writer
from megatron.training import get_model
from megatron.utils import get_ltor_masks_and_position_ids
from models.reward_model import batch_padded_tokenize_data, RewardModel

from chatlearn import RLHFMegatronModule
from chatlearn.utils import to_device
from .constants_ppo import RunningMoments, get_running_stats, reset_running_stats
from .forward_step import forward_step_helper
from .utils import tensorboard_scalar_dict

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def dump_jsonl_chinese(res, file_path, mode="w"):
    print(f"writing jsonl to : {file_path}")
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with codecs.open(file_path, mode, 'utf-8') as outfile:
        for entry in res:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')


def save_list_str(list_strs, iteration):
    args = get_args()
    inference_output_path = f"{args.log_dir}/{args.exp_name}/inference_outputs_{iteration}.json"
    Path(inference_output_path).parent.mkdir(parents=True, exist_ok=True)

    res = []
    for prompt, response in list_strs:
        k = {"query": prompt, "responses": [response], "iteration": iteration}
        res.append(k)
    dump_jsonl_chinese(res, inference_output_path, mode="a")


class RewardInference(RLHFMegatronModule):
    """RewardInference"""

    def setup(self):
        self.buffer = {}
        self.stats = {}
        # setup how you get reward. This will be wrapped in a RewardModel

        self.reward_bias = self.model_args["reward_bias"]
        self.reward_type = self.model_args.get("reward_type", "reward")

        args = get_args()
        # Set up model and load checkpoint
        model = get_model(self.model_provider, wrap_with_ddp=False)

        if args.load is not None:
            load_checkpoint(model, None, None,
                            adaptive_parallel_strategy=args.adaptive_parallel_strategy_on_checkpoint)

        assert len(model) == 1, "Above condition should have caught this"
        self.model = model[0]

        self.running = RunningMoments()
        self.per_episode_metrics = defaultdict(RunningMoments)
        self.args = args

        self.tokenizer = get_tokenizer()

        self.add_padding_config("all_token_ids_right_padded", self.tokenizer.eod_id)
        self.add_padding_config("action_start_indices", self.tokenizer.eod_id)
        self.add_padding_config("action_logprobs", 0.0)
        self.add_padding_config("action_values", 0.0)
        self.add_padding_config("action_rewards", 0.0)
        return 'ok'

    def model_provider(self, pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building GPT model ...')

        model = RewardModel(
            num_tokentypes=0,
            parallel_output=True,
            ## deprecated args for now, originally set for whether the pooler header use tenoser parallel
            pre_process=pre_process,
            post_process=post_process,
            score_dimension=1,
        )
        return model

    def normalized_and_clip(self, scores):
        if self.model_args['scale_reward'] == "running":
            if self.running.count >= 2:
                scores /= self.running.std
        clip_reward = self.model_args['cliprange_reward']
        if clip_reward:
            scores = torch.clip(scores, -clip_reward, clip_reward)
        return scores

    def get_n_gram_reward(self, tokens):

        assert len(tokens.size()) == 1, f"must be 1d {tokens}"
        penalty = 0.0
        max_repetition, average_distance = self.find_ngrams(tokens, 2)
        max_repetition_3, average_distance_3 = self.find_ngrams(tokens, 3)

        if average_distance is not None:
            # must have a repetition found
            assert max_repetition >= 2, f"{max_repetition}"
            penalty += max_repetition.item() / (average_distance)

        if average_distance_3 is not None:
            assert max_repetition_3 >= 2, f"{max_repetition_3}"

            penalty += max_repetition_3.item() / (average_distance_3)

        return -penalty

    def find_ngrams(self, x, n):
        L = x.size(0)

        if L == 0:
            return 0, None
        # Pad the input tensor with zeros at the end to ensure that we can extract
        # n-grams up to the last element of the tensor.
        padded_x = torch.cat((x, torch.zeros(n - 1, device=x.device, dtype=x.dtype)), dim=0)
        # Use the unfold method to extract all sliding windows of size n from the
        # padded input tensor.
        # The step size is 1, which means we extract n-grams with overlapping
        # elements.
        # The size of the resulting tensor is (L - n + 1, n), which contains all n-grams.
        ngrams = padded_x.unfold(0, n, 1)[:L - n + 1]
        # Count the frequency of each n-gram
        unique_ngrams, counts = torch.unique(ngrams, return_counts=True, dim=0)
        if len(unique_ngrams) == 0:
            return 0, None
        max_count_index = torch.argmax(counts)
        max_count = counts[max_count_index]

        # get the most frequent n-gram
        most_frequent_ngram = unique_ngrams[max_count_index]
        if max_count >= 2:
            indices = torch.nonzero(torch.eq(ngrams, most_frequent_ngram).all(dim=1)).view(-1)

            # if diff is less than ngram size it's overlapping then count as 1
            diff = (torch.diff(indices).float().mean() - n).item()
            diff = max(diff, 1.0)
        else:
            diff = None
        return max_count, diff

    def get_math_matching_reward(self, str_prompt, str_response, training_math_golden_reg):

        def extract_answer_qianwen(p):
            p = p.strip()
            p_num = " ".join(p.split('\n'))
            if p_num:
                p_num = re.sub(r'\([^\(\)]*[^0-9,\(\)./]+[^\(\)]*\)', '', p_num)

            p_num = re.findall(r'(-?[\d,]+)(\.[\d,]+)?(/-?[\d,]+)?(\.[\d,]+)?', p_num)

            if p_num:
                p_num = ["".join(p) for p in p_num]
                p_num = [''.join(p.split(',')) for p in p_num]
                p_num = [p for p in p_num if p]
                if p_num:
                    try:
                        ret = float(eval(p_num[-1])) # pylint: disable=eval-used
                        return ret
                    except BaseException:
                        return INVALID_ANS
                return INVALID_ANS
            return INVALID_ANS

        def extract_answer(completion):
            match = ANS_RE.search(completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                try:
                    float(match_str)
                except BaseException:
                    return INVALID_ANS
                return match_str
            else:
                return INVALID_ANS

        gold_ans = training_math_golden_reg[str_prompt]
        if "####" in str_response:
            pred_ans = extract_answer(str_response)
        else:
            pred_ans = extract_answer_qianwen(str_response)

        if pred_ans != INVALID_ANS and abs(float(pred_ans) - float(gold_ans)) < 1e-4:
            score = 1.0
        else:
            score = -1.0
        return score

    def get_all_rewards(self, action_starts, action_ends, loss_mask, all_tokens_right_padded, logprobs,
                        ref_logprobs, kl_ctl, action_tokens, list_strs):

        n = loss_mask.size(0)
        if self.args.raw_reward_coeff > 0:
            scores = self.get_raw_reward(all_tokens_right_padded, action_ends)
            if mpu.is_pipeline_last_stage():
                scores = self.args.raw_reward_coeff * scores.view(-1, 1)
            else:
                # we only need last rank results, so return None for other rank
                return
        else:
            scores = torch.zeros(n, 1, device=loss_mask.device)
        if self.args.ngram_coef > 0:
            ngram_rewards = []
            for action_token in action_tokens:
                ngram_rewards.append(self.args.ngram_coef * self.get_n_gram_reward(action_token))

            ngram_rewards = torch.tensor(ngram_rewards, device=scores.device)
            if self.args.log_interval > 0:
                self.per_episode_metrics["rewards/ngram_rewards"].update(ngram_rewards)

        if self.args.math_coef > 0:
            # Get the batch.
            math_rewards = []
            for str_prompt, str_response in list_strs:
                math_rewards.append(self.args.math_coef * self.get_math_matching_reward(str_prompt, str_response,
                                                                                        self.training_math_golden_reg))

            math_rewards = torch.tensor(math_rewards, device=scores.device)
            self.per_episode_metrics["rewards/math_rewards"].update(math_rewards)

        rewards = -kl_ctl * (logprobs - ref_logprobs)
        if self.args.log_interval > 0:
            self.stats["reward_model/kl_ctl"] = kl_ctl

        # TODO: collect accross devices to support dp
        # -1 because logprobs are logprobs of action_id[1:]!!! so it's already shifted right, to get logprob of first action, we need -1
        if self.args.log_interval > 0:
            for ix, rs in enumerate(rewards):
                assert action_ends[ix] - action_starts[ix] >= 1, f"{action_ends[ix]} {action_starts[ix]}"

            kl_rewards_for_log = [rs[action_starts[ix] - 1: action_ends[ix] - 1] / (action_ends[ix] - action_starts[ix])
                                  for
                                  ix, rs in
                                  enumerate(rewards)]
            kl_rw_sums = torch.tensor([rr.sum() for rr in kl_rewards_for_log], dtype=torch.float32,
                                      device=torch.cuda.current_device())

            self.stats["rewards/klrewards_max"] = kl_rw_sums.max()
            self.stats["rewards/klrewards_min"] = kl_rw_sums.min()
            self.per_episode_metrics["rewards/klrewards"].update(kl_rw_sums)

        if self.args.lm_coef > 0:
            lm_reward = self.args.lm_coef * ref_logprobs

            # -1 because logprobs are logprobs of action_id[1:]!!! so it's already shifted right, to get logprob of first action, we need -1
            if self.args.log_interval > 0:
                lm_rewards_for_log = [
                    rs[action_starts[ix] - 1: action_ends[ix] - 1] / (action_ends[ix] - action_starts[ix])
                    for ix, rs
                    in
                    enumerate(lm_reward)]
                lm_rewards_for_log_sums = torch.tensor([rr.sum() for rr in lm_rewards_for_log], dtype=torch.float32,
                                                       device=torch.cuda.current_device())

                self.stats["rewards/lmrewards_max"] = lm_rewards_for_log_sums.max()
                self.stats["rewards/lmrewards_min"] = lm_rewards_for_log_sums.min()
                self.per_episode_metrics["rewards/lmrewards"].update(lm_rewards_for_log_sums)

            rewards += lm_reward

        # -1 because logprobs are logprobs of action_id[1:]!!! so it's already shifted right, to get logprob of first action, we need -1
        rewards = [rs[action_starts[ix] - 1: action_ends[ix] - 1] / (action_ends[ix] - action_starts[ix]) for ix, rs in
                   enumerate(rewards)]

        # Compute rewards
        all_rewards = [None] * n

        for ix in range(n):
            rs = rewards[ix]
            if len(rs) == 0:
                rs = torch.tensor([0.0])
            rs[-1] += scores[ix].cpu().item()

            if self.args.ngram_coef > 0:
                rs[-1] += ngram_rewards[ix].cpu().item()
            if self.args.math_coef > 0:
                rs[-1] += math_rewards[ix].cpu().item()

            all_rewards[ix] = rs

        all_rw_means = torch.tensor([rr.sum() for rr in all_rewards], dtype=torch.float32,
                                    device=torch.cuda.current_device())
        if self.args.log_interval > 0:
            self.per_episode_metrics["rewards/all_rw_sum"].update(all_rw_means)
            self.stats["rewards/all_rw_sum_max"] = all_rw_means.max()
            self.stats["rewards/all_rw_sum_min"] = all_rw_means.min()
        return all_rewards

    def forward_step(self, data, iteration=None):
        '''
        framework source: reward_output = self.reward.forward_step(policy_output[0], ref_output[0], old_values[0])

        ref_logprobs: log probof prompt+response (batch_size, max_sequence_length-1) gather(logits, token_ids[:, 1:])
        logprobs: log probof prompt+response (batch_size, max_sequence_length-1) already shifted
        old_value: Vhead (lm hiddens) = [b, prompt+response_len] (batch_size, max_sequence_length)
        :param batch:
        :return:
        '''
        loss_mask = to_device("cuda", data["loss_mask"])
        no_padded_query_ids = to_device("cuda", data["no_padded_query_ids"])
        all_tokens_right_padded = to_device("cuda", data["all_tokens"])
        str_prompts = data["str_prompts"]
        str_outputs = data["str_outputs"]

        list_strs = [[str_prompt, str_output] for str_prompt, str_output in zip(str_prompts, str_outputs)]

        if self.args.save_inference and iteration % self.args.save_inference_interval == 0:
            if self.is_last_rank():
                # last rank save inference output
                save_list_str(list_strs, iteration)

        old_value = data["old_values"]
        ref_logprobs = data["ref_logprobs"]
        logprobs = data["logprobs"]

        if self.args.fix_kl_coef:
            kl_coef = self.args.init_kl_coef
        else:
            kl_coef = self.get("kl_coef")
            if kl_coef is None:
                kl_coef = self.args.init_kl_coef

        # "all_token_ids_right_padded": torch.tensor([[p,p,5,6,7], [p,p,p,8,9]], dtype=torch.long, device=device),
        # "action_start_indices": torch.tensor([[10,100,p,p,p], [11,p,p,p,p]], dtype=torch.long, device=device),
        # "action_logprobs": torch.randn([bs, 5], dtype=torch.float32, device=device),
        # "action_values": torch.randn([bs, 5], dtype=torch.float32, device=device),
        # "action_rewards": torch.randn([bs, 5], dtype=torch.float32, device=device),
        if self.args.log_interval > 0:
            assert ref_logprobs.size(1) + 1 == all_tokens_right_padded.size(
                1), f"{ref_logprobs.size(1)}, {all_tokens_right_padded.size(1)} "
            assert logprobs.size(1) + 1 == all_tokens_right_padded.size(
                1), f"{logprobs.size(1)}, {all_tokens_right_padded.size(1)} "
            assert old_value.size(1) == all_tokens_right_padded.size(
                1), f"{old_value.size(1)}, {all_tokens_right_padded.size(1)} "

        n = all_tokens_right_padded.shape[0]
        # if ends with a eos_token also pad, it doesn't change.
        # if stopped due to len limit, discard last token to align with rewards.
        # because reward is r(s,a) which is a state action pair starts from state,
        # thus the last unstopped token has no reward assigned and thus need to discard
        values = old_value[:, :-1]

        if self.args.loss_on_prompts:
            # because first token has no prob and serve as the first token to attend to so no loss
            starts = torch.tensor([1 for no_padded_query_id in no_padded_query_ids], dtype=torch.long)
        else:
            starts = torch.tensor([len(no_padded_query_id) for no_padded_query_id in no_padded_query_ids],
                                  dtype=torch.long)
        ends = torch.tensor([start + loss_mask[i, start:].sum() for i, start in enumerate(starts)], dtype=torch.long)
        # -1 because logprobs are logprobs of action_id[1:]!!! so it's already shifted right, to get logprob of first action, we need -1
        if self.args.log_interval > 0:
            for ix, rs in enumerate(starts):
                assert ends[ix] - rs > 0, \
                    f"no_padded_query_id: {no_padded_query_ids[ix]}. ends[ix]: {ends[ix]} starts[ix]: {rs} " \
                    + f"loss_mask[ix] {loss_mask[ix]} all_tokens_right_padded[ix]: {all_tokens_right_padded[ix, rs:]}"
        # start = query_tensors.shape[1] - 1 is because we need state's value!! so 1 step ahead
        # eg [ pad, q1, q2, q3, a1, a2, a3, pad, pad] -> ends[i] = 4
        # eg [ pad, q1, q2, q3, a1, a2, a3] -> [ pad, q1, q2, q3, a1, a2] ends[i] = 3
        # all values = value(hidden(q3, a1, a2, a3)).
        all_values = [values[ix, starts[ix] - 1: ends[ix] - 1] for ix in range(n)]  # we want states

        action_tokens = [all_tokens_right_padded[ix, starts[ix]: ends[ix]] for ix in range(n)]

        all_rewards = self.get_all_rewards(starts, ends, loss_mask, all_tokens_right_padded, logprobs,
                                           ref_logprobs, kl_coef, action_tokens, list_strs)

        # [ pad, q1, q2, q3, a1, a2, a3], logprobs= logprob[ q1, q2, q3, a1, a2, a3]
        # start = 4 - 1 = 3 ends[i] = 4  logprobs[3: 3 + 4=7] = logprob[a1, a2, a3]]
        all_logprobs = [logprobs[ix, starts[ix] - 1: ends[ix] - 1] for ix in range(n)]

        if self.args.log_interval > 0:
            for i in range(n):
                # for each traj, num states == num actions

                assert all_logprobs[i].size(0) == all_values[i].size(0) == all_rewards[i].size(0), \
                    f"all_rewards[i].size() {all_rewards[i].size(0)} all_values[i].size(0) {all_values[i].size(0)}" \
                    f"all_logprobs[i].size(0) {all_logprobs[i].size(0)}"

        if self.args.log_interval > 0 and iteration % self.args.log_interval == 0:
            self.log_each_step(iteration)

        return {"all_token_ids_right_padded": all_tokens_right_padded, "action_start_indices": starts,
                "action_logprobs": all_logprobs,
                "action_values": all_values, "action_rewards": all_rewards, "loss_mask": loss_mask}

    def log_each_step(self, iteration):
        writer = get_tensorboard_writer()
        stats_episode = get_running_stats(self.per_episode_metrics)
        stats_episode.update(self.stats)

        stats_episode["exp_scores/running_mean"] = self.running.mean
        stats_episode["exp_scores/running_std"] = self.running.std

        # RL related stats: global
        if self.is_last_rank():
            tensorboard_scalar_dict(writer, prefix=f"rewards_each/replica_id{self.replica_id}",
                                    global_step=iteration,
                                    scalar_dict=stats_episode)
        # reset runnings
        reset_running_stats(self.per_episode_metrics)

    def eval_forward(self, policy_res: dict):
        '''
        policy_res: {"all_tokens": tokens,  "str_samples": str_samples,
                "str_prompts": str_prompts, "str_outputs": str_outputs, "logprobs": all_log_probs,
                "no_padded_query_ids": no_padded_query_ids}

        output: [{"query":str_prompt, "responses": [str_output], "eval_score_dict": score_dict}, ...]
        :param policy_res:
        :return:
        '''
        str_prompts = policy_res["str_prompts"]
        str_outputs = policy_res["str_outputs"]
        all_tokens_right_padded = to_device("cuda", policy_res["all_tokens"])
        no_padded_query_ids = to_device("cuda", policy_res["no_padded_query_ids"])
        loss_mask = to_device("cuda", policy_res["loss_mask"])

        if self.args.loss_on_prompts:
            # because first token has no prob and serve as the first token to attend to so no loss
            starts = torch.tensor([1 for no_padded_query_id in no_padded_query_ids], dtype=torch.long)
        else:
            starts = torch.tensor([len(no_padded_query_id) for no_padded_query_id in no_padded_query_ids],
                                  dtype=torch.long)
        ends = torch.tensor([start + loss_mask[i, start:].sum() for i, start in enumerate(starts)], dtype=torch.long)

        list_strs = [[str_prompt, str_output] for str_prompt, str_output in zip(str_prompts, str_outputs)]

        if get_args().do_math_eval:
            # math only
            math_rewards = []
            for str_prompt, str_response in list_strs:
                final_answer_rw = self.get_math_matching_reward(str_prompt, str_response, self.test_math_golden_reg)
                if final_answer_rw == -1.0:
                    final_answer_rw = 0.0
                math_rewards.append(self.args.math_coef * final_answer_rw)

            reward_model_scores = torch.tensor(math_rewards, device="cuda")

        else:
            reward_model_scores = self.get_raw_reward(all_tokens_right_padded, ends).view(-1, 1)
        self.per_episode_metrics["eval_rewards/reward_model_scores"].update(reward_model_scores)

        reward_checkpoint = self.args.load
        reward_checkpoint_load_iteration = self.args.load_iteration

        output = []
        rewards_output = []
        for str_prompt, str_output, reward in zip(str_prompts, str_outputs, reward_model_scores):
            rw = reward.cpu().item()
            rewards_output.append(rw)

            score_dict = {reward_checkpoint: {reward_checkpoint_load_iteration: [rw]}}
            j = {"query": str_prompt, "responses": [str_output], "eval_score_dict": score_dict}
            output.append(j)

        return {"eval_jsonl": output, "rewards": rewards_output}

    def forward_step_pipeline(self, list_strs=None, all_tokens_right_padded=None, ends=None):
        self.model.eval()
        args = get_args()

        if list_strs:
            assert not all_tokens_right_padded and not ends, \
                "Expected all_tokens_right_padded=None and ends=None in forward_step_pipeline, " \
                f"but got {type(all_tokens_right_padded)} and {type(ends)}."
            input_ids, pooling_sequence_index = batch_padded_tokenize_data(list_strs, self.tokenizer, args.seq_length)
        else:
            assert all_tokens_right_padded is not None and ends is not None, \
                "Expected non-empty all_tokens_right_padded and ends in forward_step_pipeline, " \
                f"but got {type(all_tokens_right_padded)} and {type(ends)}."
            input_ids, pooling_sequence_index = all_tokens_right_padded, ends - 1

        input_ids = input_ids.cuda()
        pooling_sequence_index = pooling_sequence_index.cuda()

        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            input_ids,
            self.tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        batch_size = input_ids.size(0)

        # =============
        # Run infernece
        # =============
        with torch.no_grad():
            # logits will be meanigful only in the last pipeline stage.
            output_rewards = forward_step_helper(self.model, input_ids, position_ids, attention_mask, pooling_sequence_index,
                                            pooling=True)
            if mpu.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert output_rewards is not None
                assert batch_size == 1 or output_rewards.size(0) == batch_size
                return output_rewards

    def get_raw_reward(self, all_tokens_right_padded, ends):
        exp_score_time = time()
        scores = self.forward_step_pipeline(all_tokens_right_padded=all_tokens_right_padded, ends=ends)
        # minus the sft baseline
        if mpu.is_pipeline_last_stage():
            scores -= self.reward_bias
            if self.args.log_interval > 0:
                self.stats["time/exp_score"] = time() - exp_score_time
                self.stats["rewards/reward_model_scores_max"] = scores.max()
                self.stats["rewards/reward_model_scores_min"] = scores.min()
                self.per_episode_metrics["rewards/reward_model_scores"].update(scores)
            scores = self.normalized_and_clip(scores)
            if self.args.log_interval > 0:
                self.stats["rewards/normalized_clip_scores_max"] = scores.max()
                self.stats["rewards/normalized_clip_scores_min"] = scores.min()
                self.per_episode_metrics["rewards/normalized_clip_scores"].update(scores)
            return scores
