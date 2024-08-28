import torch
from typing import Optional

from chatlearn import DeepSpeedModule
import torch.nn.functional as F
from data.reward_dataset import RewardDataset
from .utils import DPOModel


class ReferenceModel(DPOModel):

    def forward_step(self, data, iteration=0):
        chosen_ids = data["chosen_input_ids"].squeeze(1).cuda()
        c_mask = data["chosen_attention_mask"].squeeze(1).cuda()
        reject_ids = data["reject_attention_mask"].squeeze(1).cuda()
        r_mask = data["reject_attention_mask"].squeeze(1).cuda()
        prompt_id_lens = data["extra"]
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
            )
        data.update({"reference_chosen_logps": reference_chosen_logps, "reference_rejected_logps": reference_rejected_logps})
        return data

    def build_dataset(self, data, is_eval=False):
        reward_dataset = RewardDataset(data, self.args, self.tokenizer, self.args.max_len, input_template=self.args.input_template, is_dpo=True)
        return reward_dataset
