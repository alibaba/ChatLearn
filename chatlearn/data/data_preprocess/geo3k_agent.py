"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import base64
import os
from typing import List, Dict
from io import BytesIO

import datasets
from PIL import Image

def image_to_base64(img: Image.Image) -> str:

    img = img.convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=100)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def prepare_image_content(img_list: List[Image.Image]) -> List[Dict]:
    img_content = []

    for img in img_list:
        img_content.append(
            {
                "type": "image_url",
                "image_url": f"data:image;base64,{image_to_base64(img)}"
            }
        )
        assert img_content[-1]["image_url"] is not None
    return img_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="dataset/geo3k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "hiyouga/geometry3k"

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"You must use the `calc_geo3k_reward` tool after step by step solving the question"
        r"The final answer MUST BE put in \boxed{}."
    )

    system_prompt = (
                    "You are a math expert. You are given a question and you need to solve it step by step. "
                    "Reasoning step by step before any tool call. "
                    "You should use the `calc_geo3k_reward` tool after step by step solving the question, "
                    "before generate final answer at least once and refine your answer if necessary. "
                    "Put your final answer within \\boxed{}."
                )
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            prompt = problem + " " + instruction_following
            prompt = prompt.replace("<image>", "")
            answer = example.pop("answer")
            images = example.pop("images")
            # format openai style messages
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            messages[-1]["content"] = prepare_image_content(images)+ \
                messages[-1]["content"]
            data = {
                "agent_name": "geo3k_agent",
                "agent_cfg_path": "template/agent/geo3k_eval.yaml",
                "data_source": data_source,
                "messages": messages,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # to_parquet may produce key: None in dict
    train_dataset.to_parquet(os.path.join(local_dir, "train_agent.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test_agent.parquet"))
