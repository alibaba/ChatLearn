from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from chatlearn import FSDPModule
from chatlearn.utils import to_device

from .loss_gallery import calculate_grpo_loss


REF_TAG = "ref_logprobs"
OLD_TAG = "old_logprobs"

def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

def generate_loss_mask_position_ids(tokens: torch.Tensor, prompt_token_length: list, response_token_length:list):
    # Setup attention mask by prompt token length and response token length
    loss_mask = torch.zeros_like(tokens, dtype=torch.int32, device=tokens.device)
    for i in range(len(prompt_token_length)):
        loss_mask[i, prompt_token_length[i]: prompt_token_length[i] + response_token_length[i]] = 1.0
    _, seq_len = tokens.size()
    position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0).expand_as(tokens)

    return loss_mask, position_ids

class PolicyTrainer(FSDPModule):

    def setup(self):
        super().setup()
        self._metric_prefix = "policy_trainer"

    def preprocess_data(self, data_b, trainable: bool = True):
        '''
        Data preprocess for training
        data_b: dict of tensors
        trainable: whether this batch is for training
        '''

        tokens_ = data_b["all_tokens"].long()
        prompt_token_length = data_b["prompt_token_length"]
        response_token_length = data_b["response_token_length"]
        _, ori_seq_len = tokens_.size()

        # Generate loss mask, positon ids
        loss_mask, position_ids = generate_loss_mask_position_ids(tokens_, prompt_token_length, response_token_length)

        # find max seq_len in batch
        max_len = max([prompt_len + response_len for prompt_len, response_len in zip(prompt_token_length, response_token_length)])

        # shift and cut to max_seq_len in batch
        tokens = tokens_[:, :max_len - 1].contiguous()
        labels = tokens_[:, 1 :max_len].contiguous()
        loss_mask = loss_mask[: , 1 :max_len]
        position_ids = position_ids[:, :max_len - 1]

        if trainable:
            # length of logprobs is 1 token less than seq_len
            old_logprobs = data_b[OLD_TAG][:, :max_len - 1]
            ref_logprobs = data_b[REF_TAG][:, :max_len - 1]
            inputs = {
                "all_tokens": tokens,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "advantages":data_b["advantages"],
                "prompt_token_length": prompt_token_length,
                "labels": labels,
                "old_logprobs": old_logprobs,
                "ref_logprobs": ref_logprobs,
            }
        else:
            inputs = {
                "all_tokens": tokens,
                "position_ids": position_ids,
                "labels": labels,
                "repad_size": ori_seq_len - tokens.shape[1]
            }

        for k, v in inputs.items():
            inputs[k] = to_device(torch.cuda.current_device(), v)

        return inputs

    def train_step(self, data_list):
        '''
        data_list: list of micro batchs [micro_bs0, micro_bs1]
        '''
        self.model.train()  # reset model state
        self.optimizer.zero_grad()
        pg_loss_list = []
        entropy_loss_list = []
        kl_loss_list = []
        micro_bs_num = len(data_list)
        for data_b in data_list:
            inputs = self.preprocess_data(data_b)
            output = self.model(
                input_ids=inputs["all_tokens"],
                attention_mask=None,
                position_ids=inputs["position_ids"],
                use_cache=False
            )
            logits = output.logits #.squeeze(0)
            logprobs = logprobs_from_logits(logits, inputs["labels"])
            loss = calculate_grpo_loss(
                log_probs=logprobs,
                old_log_probs=inputs["old_logprobs"],
                advantages=inputs["advantages"],
                diff_clip_ratio=self.module_args.args_dict.get("diff_clip_ratio", 10),
                pos_clip_ratio=self.module_args.args_dict.get("pos_clip_ratio", 0.2),
                negative_clip_ratio=self.module_args.args_dict.get("negative_clip_ratio", 0.2),
                final_clip_ratio=self.module_args.args_dict.get("final_clip_ratio", 3)
                )
            
            pg_loss = torch.masked_select(loss, inputs["loss_mask"].bool())
            pg_loss_mean = torch.mean(pg_loss) / micro_bs_num
            pg_loss_mean.backward()
            pg_loss_list.append(pg_loss)

            # kl loss
            kl = inputs['ref_logprobs'] - logprobs
            ratio = torch.exp(kl)
            assert not torch.isinf(ratio).any(), "kl loss ratio has inf values"
            assert not torch.isnan(ratio).any(), "kl loss ratio has nan values"
            kld = (ratio - kl - 1).contiguous()
            kl_loss = torch.clamp(kld, min=-10, max=10)
            kl_loss = torch.masked_select(kl_loss, inputs["loss_mask"].bool())
            kl_loss_list.append(kl_loss)

            # entropy loss
            entropy_loss = torch.masked_select(-logprobs, inputs["loss_mask"].bool())
            entropy_loss_list.append(entropy_loss)
        grad_norm = self.model.clip_grad_norm_(max_norm=self.module_args.args_dict.get("grad_clip", 1)).detach().item()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # collect metric
        pg_loss = torch.mean(torch.cat(pg_loss_list)).detach().item()
        kl_loss = torch.mean(torch.cat(kl_loss_list)).detach().item()
        entropy_loss = torch.mean(torch.cat(entropy_loss_list)).detach().item()

        train_stats = {
            "pg_loss": pg_loss,
            "kl_loss": kl_loss,
            "entropy_loss": entropy_loss,
            "grad_norm": grad_norm
        }
        self._metric_list.append(train_stats)

    def forward_step(self, data):
        inputs =  self.preprocess_data(data, trainable=False)
        with torch.no_grad():
            output = self.model(
                input_ids=inputs['all_tokens'],
                attention_mask=None,
                position_ids=inputs['position_ids'],
                use_cache=False
            )
            logprobs = logprobs_from_logits(output.logits, inputs['labels'])
            # Repad logprobs to max_seq_len to allow concatenation
            logprobs = F.pad(logprobs, (0, inputs['repad_size']), mode='constant', value=0)
        tag = OLD_TAG
        if OLD_TAG in data.keys():
            tag = REF_TAG
        data.update({tag: logprobs})
        return data