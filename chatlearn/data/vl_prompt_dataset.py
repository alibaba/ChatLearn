"""prompt dataset"""

from collections import defaultdict
from typing import List, Dict

from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor
from chatlearn.data.vision_utils import process_image, process_video, postprocess_data
import re

class PromptPipeline(Dataset):
    """
    process vl data in this format
    {
        "images": [PIL.Image]
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
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
    self.data format
    {"input_ids": prompt_ids, "prompt": prompt}
    """

    def __init__(
        self,
        data_list: List[Dict],
        seq_length: int,
        tokenizer: AutoTokenizer = None,
        processor: AutoProcessor = None,
        enable_thinking=False
    ):  # pylint: disable=super-init-not-called
        super().__init__()

        self.tokenizer = tokenizer
        self.processor = processor

        # TODO default key for input_data
        self.prompt_key = "prompt"
        self.image_key = "images"
        self.video_key = "videos"
        self.data = []

        for data_item in data_list:
            messages = self._build_messages(data_item)

            model_inputs = {}
            
            assert self.processor is not None

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # multi_modal_data = {}

            images = None
            if self.image_key in data_item and data_item.get(self.image_key, None) is not None:
                images = [process_image(image) for image in data_item.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                # multi_modal_data["image"] = images

            videos = None
            if self.video_key in data_item and data_item.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in data_item.pop(self.video_key)]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                # multi_modal_data["video"] = [video.numpy() for video in videos]

            # TODO support video. Only images are supported now.
            multi_modal_data = {'image':images}
            mm_processor_kwargs = {'fps': []}

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
        
            # model_inputs = self.processor(text=[raw_prompt], images=None, videos=None, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            image_grid_thw = model_inputs.get("image_grid_thw")
            pixel_values = model_inputs.get("pixel_values")

            # text only for vllm
            # model_inputs = self.processor(text=[raw_prompt], images=None, videos=None, return_tensors="pt")
            raw_input_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

            # input_ids, attention_mask = postprocess_data(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     max_length=seq_length,
            #     pad_token_id=self.tokenizer.pad_token_id,
            #     left_pad=True
            # )

            if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                from chatlearn.models.patches.transformers.qwen2_vl import get_rope_index
                position_ids, rope_deltas = get_rope_index(
                        self.processor,
                        input_ids=input_ids,
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                        attention_mask=attention_mask,
                )
                # position_ids = [
                #     get_rope_index(
                #         self.processor,
                #         input_ids=input_ids[0],
                #         image_grid_thw=model_inputs.get("image_grid_thw"),
                #         video_grid_thw=model_inputs.get("video_grid_thw"),
                #         second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                #         attention_mask=attention_mask[0],
                #     )
                # ]  # (1, 3, seq_len)

            data_source = data_item.get("data_source", "")
            ground_truth = data_item["reward_model"]["ground_truth"]

            processed_data = {
                "input_ids": input_ids[0].tolist(),
                "raw_input_ids": raw_input_ids,
                "prompt": raw_prompt,
                "attention_mask": attention_mask[0].tolist(),
                "position_ids": position_ids.squeeze().tolist(),
                "rope_deltas": rope_deltas,
                "data_source": data_source,
                "ground_truth": ground_truth,
                "multi_modal_data": multi_modal_data,
                "mm_processor_kwargs": mm_processor_kwargs,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw
            }

            if seq_length > len(input_ids[0]):
                self.data.append(processed_data)
    
    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, ix: int):
        return self.data[ix]

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn(self, samples):
        collate_dict = defaultdict(list)

        # Loop over the samples and append each tensor value to the corresponding list
        for sample in samples:
            for key in sample.keys():
                collate_dict[key].append(sample[key])

        # Return the collate_dict
        return collate_dict
