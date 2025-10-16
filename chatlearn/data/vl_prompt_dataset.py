"""prompt dataset"""

from typing import List, Dict

from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# from chatlearn.models.patches.transformers.qwen2_5_vl_patch import get_rope_index


class PromptPipeline(Dataset):
    """
    Input data_list: List[Dict])
    {
        "data_source": data_source,
        "messages": openai-style messages List,
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'answer': answer_raw,
            "question": question_raw,
        }
    }

    Output self.data: List[Dict])
    {
        "raw_input_ids": List, # only text input_ids for vllm inference
        "input_ids": List, # input_ids with image pad for model forward/sglang inference
        "prompt_token_length": int, # len(input_ids)
        "prompt": String,
        "position_ids": List[List], # [3, token_length]
        "data_source": String,
        "ground_truth": String,
        "multi_modal_data": {'image':[PIL.Image]}, # for vllm inference
        "mm_processor_kwargs": {'fps':[]}, # used for video useless now
        "pixel_values": Tensor, # [grid_num, pixel_num]
        "image_grid_thw": Tensor, # [1,3] 3 means t,h,w
    }
    """
    def __init__(
        self,
        data_list: List[Dict],
        max_prompt_tokens_length: int,
        tokenizer: AutoTokenizer = None,
        processor: AutoProcessor = None,
        enable_thinking=False,
        raw_chat=False
    ):  # pylint: disable=super-init-not-called
        super().__init__()

        self.tokenizer = tokenizer
        self.processor = processor

        self.data = []
        self.max_prompt = 0

        for data_item in data_list:
            messages = data_item.get("messages")
            data_source = data_item.get("data_source", "")
            ground_truth = data_item["reward_model"]["ground_truth"]
            agent_name = data_item.get("agent_name", None)
            agent_cfg_path = data_item.get("agent_cfg_path", None)
            processed_data = {
                "data_source": data_source,
                "ground_truth": ground_truth,
                "agent_name": agent_name,
                "agent_cfg_path": agent_cfg_path
            }
            for message in messages:
                message['content'] = [
                {k: v for k, v in item.items() if v is not None}
                for item in message['content']
            ]
            if not raw_chat:

                model_inputs = {}

                assert self.processor is not None

                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
                images, videos = process_vision_info(messages)

                # TODO support video. Only images are supported now.
                multi_modal_data = {'image':images}
                mm_processor_kwargs = {'fps': []}

                model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
                input_ids = model_inputs.pop("input_ids")
                attention_mask = model_inputs.pop("attention_mask")
                image_grid_thw = model_inputs.get("image_grid_thw")
                pixel_values = model_inputs.get("pixel_values")

                # text only input_ids for vllm
                raw_input_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
                # get position_ids used for sequence packing
                # position_ids, _ = get_rope_index(
                #     self.processor,
                #     input_ids=input_ids,
                #     image_grid_thw=model_inputs.get("image_grid_thw"),
                #     video_grid_thw=model_inputs.get("video_grid_thw"),
                #     second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                #     attention_mask=attention_mask,
                # )

                # for vl model, raw_input_ids is only text input_ids for vllm inference
                # input_ids is used for model forward_step and sglang inference (with image pad)
                # sglang support both input_ids and raw_input_ids but to merge in all_tokens, input_ids is used
                processed_data.update({
                    "raw_input_ids": raw_input_ids,
                    "input_ids": input_ids[0].tolist(),
                    "prompt_token_length": len(input_ids[0].tolist()),
                    "prompt": raw_prompt,
                    # "position_ids": position_ids.squeeze().tolist(),
                    "multi_modal_data": multi_modal_data,
                    "mm_processor_kwargs": mm_processor_kwargs,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                    "attention_mask": attention_mask[0].tolist(),
                })
                if len(input_ids[0]) > self.max_prompt:
                    self.max_prompt = len(input_ids[0])

                if max_prompt_tokens_length > len(input_ids[0]):
                    self.data.append(processed_data)
            else:
                # used in agent module
                processed_data.update(
                    {"messages": messages}
                )
                self.data.append(processed_data)

        self.valid_ratio = len(self.data) / len(data_list)

    def __getitem__(self, ix: int):
        return self.data[ix]

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn(self, samples):
        return samples
