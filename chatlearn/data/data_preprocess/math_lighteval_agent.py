"""
Preprocess the math dataset to json format for agent task
"""

import os
import argparse

import datasets


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--local_dir", default="~/data/math")

    args = parser.parse_args()

    data_source = "DigitalLearningGmbH/MATH-lighteval"
    data_dir = (
        "DigitalLearningGmbH/MATH-lighteval"
        if args.input_dir is None
        else args.input_dir
    )

    dataset = datasets.load_dataset(data_dir)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        "Let's think step by step and output the final answer within \\boxed{}."
        "You must use the `mathlighteval_reward` tool after step by step solving the question"
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            answer = example.pop("solution")
            solution = extract_solution(answer)
            system_prompt = (
                            "You are a math expert. You are given a question and you need to solve it step by step. "
                            "Reasoning step by step before any tool call. "
                            "You should use the `mathlighteval_reward` tool after step by step solving the question, "
                            "before generate final answer at least once and refine your answer if necessary. "
                            "Put your final answer within \\boxed{}."
                        )
            data = {
                "agent_name": "matheval_agent",
                "agent_cfg_path": "template/agent/math_eval.yaml",
                "data_source": data_source,
                "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir

    train_dataset.to_json(os.path.join(local_dir, "train_agent.json"))
    test_dataset.to_json(os.path.join(local_dir, "test_agent.json"))
