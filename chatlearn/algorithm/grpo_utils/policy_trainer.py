# Copyright 2025 Alibaba Group Holding Limited. All Rights Reserved.
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
"""FSDP Trainer"""

import math
from contextlib import nullcontext
from typing import List, Dict, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from flash_attn.bert_padding import pad_input, unpad_input

from chatlearn import FSDPModule
from chatlearn.utils import to_device
from chatlearn.utils.communication_op import gather, get_sp_parallel_group
from chatlearn.runtime.decorator import timeit, compute_decorator, monitor_error
from chatlearn.algorithm.grpo_utils.loss_gallery import calculate_grpo_loss
from chatlearn.algorithm.grpo_utils.trainer_utils import (logprobs_from_logits,
                            entropy_from_logits_with_chunking,
                            sp_split,
                            generate_loss_mask_position_ids,
                            split_microbatch,
                            batching,
                            split_and_unpadding)

class PolicyTrainer(FSDPModule):
    """policy trainer"""
    def setup(self):
        super().setup()
        self._metric_prefix = "policy_trainer"

    def split_and_padding(self, tokens: torch.Tensor, position_ids: torch.Tensor):
        """
        Preprocess tokens and position_ids given sp_size.
        Tokens and position_ids will be:
            - padded to integer multiple of sp_size
            - split into sp_size splits on seq_len dim
        Args:
            tokens (torch.Tensor): tokens with shape [bsz, seqlen]
            position_ids (torch.Tensor): position_ids with shape [bsz, seqlen]
        Returns:
            tokens (torch.Tensor): tokens for current local rank
            labels (torch.Tensor): tokens for current local rank
            position_ids (torch.Tensor): tokens for current local rank
            pad_size (int): pad_size for padding
        """
        # Pad inputs to ensure seq_len is divisible by sp_size
        valid_len = tokens.shape[1]
        pad_size = math.ceil(valid_len / self.sp_size) * self.sp_size - valid_len

        # Pad tokens and position_ids, transfomers only need this two inputs
        tokens = F.pad(tokens, (0, pad_size), value=self.tokenizer.pad_token_id)
        position_ids = F.pad(position_ids,(0, pad_size), value=valid_len)

        labels = torch.roll(tokens, shifts=-1, dims=1)

        # Split tensor by sp_size
        sp_group = get_sp_parallel_group()
        sp_local_rank = dist.get_rank(sp_group)
        tokens = sp_split(input_tensor=tokens, split_dim=1, sp_size=self.sp_size, sp_local_rank=sp_local_rank)
        labels = sp_split(input_tensor=labels, split_dim=1, sp_size=self.sp_size, sp_local_rank=sp_local_rank)
        return tokens, labels, position_ids, pad_size

    def preprocess_data_list(self, data_list: List[Dict[str, Any]], training: bool):
        # compute response length sum in train global batch size for token-wise pg loss
        response_token_length_total = torch.tensor(sum(data["response_token_length"] for data in data_list)).cuda() / self.sp_size
        dist.all_reduce(response_token_length_total, op=dist.ReduceOp.SUM)

        # Split minibathc into microbatchs
        if self.packing:
            microbatch_list = split_microbatch(data_list=data_list, max_train_token=self.max_token_in_seq, packing=self.packing)
        else:
            if training:
                microbatch_size = self.train_micro_batch_size
            else:
                microbatch_size = self.generate_micro_batch_size
            microbatch_list = split_microbatch(data_list=data_list, micro_batch_size=microbatch_size, packing=self.packing)

        # Batching
        data_list = [batching(data_b) for data_b in microbatch_list]

        data_after_process = []
        for data_b in data_list:
            data_obj = {}
            tokens_ = data_b["all_tokens"].long()
            prompt_token_length = data_b["prompt_token_length"]
            response_token_length = data_b["response_token_length"]
            ori_batch_size, ori_seq_len = tokens_.size()
            attn_mask, loss_mask, position_ids = generate_loss_mask_position_ids(tokens_, prompt_token_length, response_token_length)
            indices = None
            if self.packing:
                # Packing data into one batch
                tokens_, indices, *_ = unpad_input(tokens_.unsqueeze(-1).cuda(), attn_mask.cuda())
                tokens_ = tokens_.permute(1,0).cpu() # For compatible with transformers

                position_ids, *_ = unpad_input(position_ids.unsqueeze(-1).cuda(), attn_mask.cuda())
                position_ids = position_ids.permute(1, 0).cpu() # For compatible with transformers

            if self.sp_size > 1:
                # Pad inputs to ensure seq_len is divisible by sp_size
                tokens, labels, position_ids, pad_size = self.split_and_padding(tokens_, position_ids)
            else:
                pad_size = 0
                tokens = tokens_
                labels = torch.roll(tokens, shifts=-1, dims=1)
            data_obj.update(
                {
                    "all_tokens": tokens,
                    "position_ids": position_ids,
                    "labels": labels,
                    "ori_seq_len": ori_seq_len,
                    "indices": indices,
                    "ori_batch_size": ori_batch_size,
                    "sample_ids": data_b["id_in_list"],
                    "attention_mask": attn_mask,
                    "pad_size": pad_size,
                }
            )
            if training:
                loss_mask = torch.roll(loss_mask, shifts=-1, dims=1)
                # The last token should always be masket out
                loss_mask[:, -1] = 0
                data_obj.update(
                    {
                        "loss_mask": loss_mask,
                        "old_logprobs": data_b["old_logprobs"],
                        "ref_logprobs": data_b["ref_logprobs"],
                        "advantages": data_b["advantages"],
                    }
                )
            data_after_process.append(data_obj)
        return response_token_length_total, data_after_process

    @monitor_error()
    @compute_decorator(trainable=True, rollout=False)
    @timeit()
    def train_step(self, data_list: List[Dict[str, Any]], **kwargs): # pylint: disable=unused-argument
        """
        data_list: list of micro batchs [micro_bs0, micro_bs1]
        """
        self.model.train()  # reset model state
        self.optimizer.zero_grad()
        pg_loss_list = []
        entropy_loss_list = []
        kl_loss_list = []
        sp_group = get_sp_parallel_group()
        response_token_length_total, data_list = self.preprocess_data_list(data_list=data_list, training=True)
        for inputs in data_list:
            for k, v in inputs.items():
                inputs[k] = to_device(torch.cuda.current_device(), v)
            output = self.model(
                input_ids=inputs["all_tokens"],
                attention_mask=None,
                position_ids=inputs["position_ids"],
                use_cache=False,
            )
            logprobs = logprobs_from_logits(output.logits, inputs["labels"])

            # save memory while not use entropy in loss
            entropy_context = nullcontext() if self.module_args.entropy_coef > 0 else torch.no_grad()
            with entropy_context:
                entropy = entropy_from_logits_with_chunking(output.logits)

            if sp_group is not None:
                logprobs = gather(
                    input_tensor=logprobs, sp_group=sp_group, gather_dim=1
                )
                entropy = gather(
                    input_tensor=entropy, sp_group=sp_group, gather_dim=1
                )
            if self.packing:
                # Recover packing sequence
                logprobs = pad_input(
                    logprobs[0, :logprobs.shape[1] - inputs['pad_size']].unsqueeze(-1),
                    inputs['indices'],
                    inputs['ori_batch_size'],
                    inputs['ori_seq_len']).squeeze(-1)
                entropy = pad_input(
                    entropy[0, :entropy.shape[1] - inputs['pad_size']].unsqueeze(-1),
                    inputs['indices'],
                    inputs['ori_batch_size'],
                    inputs['ori_seq_len']).squeeze(-1)
            else:
                logprobs_len = logprobs.shape[1]
                logprobs = F.pad(logprobs, (0, inputs['ori_seq_len'] - logprobs_len), mode='constant', value=0)
                entropy = F.pad(entropy, (0, inputs['ori_seq_len'] - logprobs_len), mode='constant', value=0)
            loss = calculate_grpo_loss(
                log_probs=logprobs,
                old_log_probs=inputs["old_logprobs"],
                advantages=inputs["advantages"],
                diff_clip_ratio=self.module_args.get("diff_clip_ratio", 10),
                pos_clip_ratio=self.module_args.get("pos_clip_ratio", 0.2),
                neg_clip_ratio=self.module_args.get("neg_clip_ratio", 0.2),
                final_clip_ratio=self.module_args.get("final_clip_ratio", 3),
            )

            pg_loss = torch.masked_select(loss, inputs["loss_mask"].bool())

            # entropy loss
            entropy_loss = torch.masked_select(entropy, inputs["loss_mask"].bool())
            entropy_loss_mean = torch.sum(entropy_loss) / response_token_length_total * self.fsdp_size

            # kl loss
            kl = inputs["ref_logprobs"] - logprobs
            kl = torch.masked_select(kl, inputs["loss_mask"].bool())
            ratio = torch.exp(kl)
            assert not torch.isinf(ratio).any(), "kl loss ratio has inf values"
            assert not torch.isnan(ratio).any(), "kl loss ratio has nan values"
            kld = (ratio - kl - 1).contiguous()
            kl_loss = torch.clamp(kld, min=-10, max=10)
            kl_loss_mean = torch.sum(kl_loss) / response_token_length_total * self.fsdp_size

            # compute backward loss
            pg_loss_mean = torch.sum(pg_loss) / response_token_length_total * self.fsdp_size
            total_loss = pg_loss_mean
            if self.module_args.entropy_coef > 0:
                total_loss = total_loss - self.module_args.entropy_coef * entropy_loss_mean
            if self.module_args.kl_coef > 0:
                total_loss = total_loss + self.module_args.kl_coef * kl_loss_mean
            total_loss.backward()

            pg_loss_list.append(pg_loss.detach())
            entropy_loss_list.append(entropy_loss.detach())
            kl_loss_list.append(kl_loss.detach())

        # refs to https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#gradient-clipping-and-optimizer-with-dtensor
        # but results seems not right in torch 2.6.0+cu124
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.module_args.optimizer.clip_grad).detach().item()
        grad_norm = self.fsdp2_clip_grad_norm_(self.model.parameters(), max_norm=self.module_args.optimizer.clip_grad).detach().item()

        self.optimizer.step()
        self.optimizer.zero_grad()

        # collect metric
        pg_loss = (torch.sum(torch.cat(pg_loss_list)) / response_token_length_total * self.fsdp_size).item()
        kl_loss = torch.mean(torch.cat(kl_loss_list)).item()
        entropy_loss = torch.mean(torch.cat(entropy_loss_list)).item()

        train_stats = {
            "pg_loss": pg_loss,
            "kl_loss": kl_loss,
            "entropy_loss": entropy_loss,
            "grad_norm": grad_norm,
        }
        self._metric_list.append(train_stats)

    @monitor_error()
    @compute_decorator(trainable=False, rollout=False)
    @timeit()
    def forward_step(self, data: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]: # pylint: disable=unused-argument,arguments-differ
        _, data_list = self.preprocess_data_list(data_list=data, training=False)
        tag = "old_logprobs" if self.trainable else "ref_logprobs"
        # Logprobs holder
        for inputs in data_list:
            for k, v in inputs.items():
                inputs[k] = to_device(torch.cuda.current_device(), v)
            with torch.no_grad():
                output = self.model(
                    input_ids=inputs['all_tokens'],
                    attention_mask=None,#inputs['attention_mask'],
                    position_ids=inputs['position_ids'],
                    use_cache=False
                )
                sp_group = get_sp_parallel_group()
                logprobs = logprobs_from_logits(output.logits, inputs["labels"])
                if sp_group is not None:
                    logprobs = gather(input_tensor=logprobs, sp_group=sp_group, gather_dim=1)
                # Repad logprobs to max_seq_len to allow concatenation
                if self.packing:
                    # Recover packing sequence
                    logprobs = pad_input(
                        logprobs[0, :logprobs.shape[1] - inputs['pad_size']].unsqueeze(-1),
                        inputs['indices'],
                        inputs['ori_batch_size'],
                        inputs['ori_seq_len']
                    ).squeeze(-1)
                else:
                    logprobs_len = logprobs.shape[1]
                    # Unpad
                    logprobs = F.pad(logprobs, (0, inputs['ori_seq_len'] - logprobs_len), mode='constant', value=0)
            # Turn logprobs tensor into list of tensors
            logprobs_tensor_list = split_and_unpadding(logprobs, inputs['attention_mask'])
            for sample_id, logprob in zip(inputs['sample_ids'], logprobs_tensor_list):
                data[sample_id].update({tag: logprob})

        return data
