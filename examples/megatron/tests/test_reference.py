import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn.functional as F
from megatron.tokenizer.tokenizer import _GPT2ZHBPETokenizer
from ref_policy_inference_megatron import PolicyReference as ReferenceModel

import rlhf
from rlhf import Engine
from utils.utils import get_loss_mask, read_jsonl


def get_all_token_ids(prompt_res_pair_batch):
    vocab_file = "/mnt/shared/Group-m6/xianyan.xianyanjia/QWen/gpt2-zhcn3-v4.json"
    merge_file = "/mnt/shared/Group-m6/xianyan.xianyanjia/QWen/gpt2-zhcn3-v4.bpe"

    tokenizer = _GPT2ZHBPETokenizer(vocab_file, merge_file)

    prompt_strs = [p['query'] for p in prompt_res_pair_batch]
    res_strs = [p['response'] for p in prompt_res_pair_batch]

    prompt_encodings = [tokenizer.tokenize(prompt_str)[:max_prompt_length] for prompt_str in prompt_strs]
    full_encodings = [(prompt_encodings[i] + tokenizer.tokenize(res_str))[:max_length] for i, res_str in
                      enumerate(res_strs)]

    no_padded_query_ids = [torch.tensor(p_encoding, dtype=torch.long, device=torch.cuda.current_device())
                           for
                           p_encoding in prompt_encodings]

    # get padded all token_ids
    all_token_ids_no_pad = [torch.tensor(all_encoding, dtype=torch.long, device=torch.cuda.current_device())
                            for
                            all_encoding in full_encodings]

    all_length = [len(all_token)
                  for all_token in all_token_ids_no_pad]
    # Get the max prompts length.
    max_len = max(all_length)
    all_token_ids = [
        F.pad(
            all_token,
            (0, max_len - len(all_token)),
            value=tokenizer.eod,  # just pad_token_id
        )
        for all_token in all_token_ids_no_pad
    ]
    all_token_ids = torch.vstack(all_token_ids).to(torch.cuda.current_device())
    assert all_token_ids.size(1) == max_len, "pad to the query_size + max generate size"

    starts = torch.tensor([len(q) for q in no_padded_query_ids], dtype=torch.long, device=all_token_ids.device)

    loss_mask = get_loss_mask(all_token_ids, tokenizer.eod, starts)

    ends = torch.tensor([start + loss_mask[i, start:].sum() for i, start in enumerate(starts)], dtype=torch.long)

    return all_token_ids, starts, ends


def plot_promptvsaction(prompt_mean_logprobs, action_mean_logprobs, out_fig_fp):
    # Creating a figure and axis
    fig, ax = plt.subplots()

    # Plotting the lists
    ax.plot(prompt_mean_logprobs, action_mean_logprobs, 'o')

    # Adding labels and title
    ax.set_xlabel('prompt_mean_logprobs')
    ax.set_ylabel('action_mean_logprobs')
    ax.set_title('prompt_mean_logprobs vs. action_mean_logprobs')

    # Saving the figure
    fig.savefig(out_fig_fp)


if __name__ == "__main__":
    rlhf.init()

    policy = ReferenceModel("reference")
    engine = Engine(policy)
    engine.setup()
    model = engine.models[0].replicas[0]
    # all_texts = {"prompts": ["who is the father of father: the father of the father is farther", "who is the father of father222: the father222 of the father2222 is farther"]}
    # train_prompt_responses = [{'query':"What is the greatest common factor of $20 !$ and $200,\\!000$?  (Reminder: If $n$ is a positive integer, then $n!$ stands for the product $1\\cdot 2\\cdot 3\\cdot \\cdots \\cdot (n-1)\\cdot n$.)", 'response': "The prime factorization of $200,000$ is $2^6 \\cdot 5^5$"}]
    train_prompt_responses = read_jsonl(
        "/mnt/shared/Group-m6/tianhang_zhu/chatgpt_api_v2/chatgpt_api_v2/all_rlhf_data/PRM800/prm800k/prm800k/data/calibration_test_test_phase2.jsonl")
    train_prompt_responses = train_prompt_responses[-2:]
    print(f"train_prompt_responses len: {len(train_prompt_responses)}")
    max_prompt_length = 512
    max_length = 1024
    batch_size = 4

    rating2logprob = {1: {"prompt_mean_logprobs": [], "action_mean_logprobs": []},
                      0: {"prompt_mean_logprobs": [], "action_mean_logprobs": []},
                      -1: {"prompt_mean_logprobs": [], "action_mean_logprobs": []}}

    for start in range(0, len(train_prompt_responses), batch_size):
        end = min(start + batch_size, len(train_prompt_responses))
        prompt_res_pair_batch = train_prompt_responses[start:end]
        ratings = [p["response_rating"] for p in prompt_res_pair_batch]
        all_tokens, starts, ends = get_all_token_ids(prompt_res_pair_batch)

        inputs = {"all_tokens": all_tokens}
        # print(f"prompt_res_pair_batch: {prompt_res_pair_batch}")
        print(f"inputs: {inputs}")
        res = model.forward_step(inputs)
        res = rlhf.get(res)[0]
        print(f"res: {res}")
        action_logprobs = [rs[starts[ix] - 1: ends[ix] - 1] for
                           ix, rs
                           in
                           enumerate(res["ref_logprobs"])]

        mean_action_logprob = [al.mean().cpu().item() for al in action_logprobs]

        print(f"mean_action_logprob: {mean_action_logprob}")
        prompt_logprobs = [rs[: starts[ix] - 1] for
                           ix, rs
                           in
                           enumerate(res["ref_logprobs"])]
        mean_prompt_logprobs = [al.mean().cpu().item() for al in prompt_logprobs]

        print(f"mean_prompt_logprobs: {mean_prompt_logprobs}")

        print(f"ratings: {ratings}")
        for i, actionlogprob in enumerate(mean_action_logprob):
            rating2logprob[ratings[i]]["action_mean_logprobs"].append(mean_action_logprob[i])
            rating2logprob[ratings[i]]["prompt_mean_logprobs"].append(mean_prompt_logprobs[i])

        print(f"1 action mean logprobs: {numpy.mean(rating2logprob[1]['action_mean_logprobs'])}")
        print(f"0 action mean logprobs: {numpy.mean(rating2logprob[0]['action_mean_logprobs'])}")
        print(f"-1 action mean logprobs: {numpy.mean(rating2logprob[-1]['action_mean_logprobs'])}")

        for rating in [1, 0, -1]:
            output_fp = f"/mnt/shared/Group-m6/tianhang_zhu/latest_rlhf_0606/results/promptlogprobvsactionlogprob_rating_{rating}.png"
            plot_promptvsaction(rating2logprob[rating]['prompt_mean_logprobs'],
                                rating2logprob[rating]['action_mean_logprobs'], output_fp)
