# pylint: skip-file
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from flash_attn.bert_padding import pad_input, unpad_input

from chatlearn import FSDPModule
from chatlearn.utils import to_device
from chatlearn.utils.communication_op import gather, get_sp_parallel_group
from .loss_gallery import calculate_grpo_loss
from .trainer_utils import (logprobs_from_logits,
                            entropy_from_logits_with_chunking,
                            sp_split,
                            generate_loss_mask_position_ids)
from .packing_utils import regroup_data_packing


REF_TAG = "ref_logprobs"
OLD_TAG = "old_logprobs"


class PolicyTrainer(FSDPModule):
    """policy trainer"""

    def setup(self):
        super().setup()
        self._metric_prefix = "policy_trainer"

    def preprocess_data_list(self, data_list, training:bool):
        # compute response length sum in train global batch size for token-wise pg loss
        response_token_length_total = torch.tensor(sum(sum(data["response_token_length"]) for data in data_list)).cuda() / self.sp_size
        dist.all_reduce(response_token_length_total, op=dist.ReduceOp.SUM)

        minibatch_size_per_rank = len(data_list) * data_list[0]["all_tokens"].size(0)
        if self.packing:
            # When packing is enabled, data_list will only contain one microbatch
            # Microbatch will be regrouped
            if not training:
                regroup_keywords = ["all_tokens", "prompt_token_length", "response_token_length"]
            else:
                regroup_keywords = ["all_tokens", "prompt_token_length", "response_token_length", "advantages", REF_TAG, OLD_TAG]
            data_list = regroup_data_packing(data_list, regroup_keywords, self.max_token_in_seq)

        data_after_process = []
        for data_b in data_list:
            tokens_ = data_b["all_tokens"].long()
            prompt_token_length = data_b["prompt_token_length"]
            response_token_length = data_b["response_token_length"]
            ori_batch_size, ori_seq_len = tokens_.size()
            if self.packing:
                # Packing data into one batch
                attn_mask, loss_mask, position_ids = generate_loss_mask_position_ids(tokens_, prompt_token_length, response_token_length)
                tokens_, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(tokens_.unsqueeze(-1).cuda(), attn_mask.cuda())
                tokens_ = tokens_.permute(1,0).cpu() # For compatible with transformers

                position_ids, *_ = unpad_input(position_ids.unsqueeze(-1).cuda(), attn_mask.cuda())
                position_ids = position_ids.permute(1, 0).cpu() # For compatible with transformers
                # Pad tokens_ to ensure max valid token length meets sp requirements
                pad_size = 0
                if self.sp_size > 1:
                    # Pad inputs to ensure seq_len is divisible by sp_size
                    valid_len = tokens_.shape[1]
                    pad_size = math.ceil(valid_len / self.sp_size) * self.sp_size - valid_len

                    # Pad tokens and position_ids, transfomers only need this two inputs
                    tokens = F.pad(tokens_, (0, pad_size), value=self.tokenizer.pad_token_id)
                    position_ids = F.pad(position_ids,(0, pad_size), value=pad_size)

                    labels = torch.roll(tokens, shifts=-1, dims=1)

                    # Split tensor by sp_size
                    sp_group = get_sp_parallel_group()
                    sp_local_rank = dist.get_rank(sp_group)
                    tokens = sp_split(input_tensor=tokens, split_dim=1, sp_size=self.sp_size, sp_local_rank=sp_local_rank)
                    labels = sp_split(input_tensor=labels, split_dim=1, sp_size=self.sp_size, sp_local_rank=sp_local_rank)
                else:
                    tokens = tokens_
                    labels = torch.roll(tokens, shifts=-1, dims=1)
                #micro_batch_seqlen = [prompt_len + response_len for prompt_len, response_len in zip(prompt_token_length, response_token_length)]
                #attention_mask = prepare_packing_attn_mask(micro_batch_seqlen, dtype=torch.get_default_dtype(), pad_size=pad_size)
                if not training:
                    data_after_process.append(
                        {
                            "all_tokens": tokens,
                            "position_ids": position_ids,
                            "labels": labels,
                            "ori_seq_len": ori_seq_len,
                            "ori_batch_size": ori_batch_size,
                            "indices": indices,
                            "bin_ids": data_b["bin_ids"],
                            "bin_seqlen": data_b["bin_seqlen"],
                            "pad_size": pad_size,
                        }
                    )
                else:
                    loss_mask = torch.roll(loss_mask, shifts=-1, dims=1)
                    # The last token should always be masket out
                    loss_mask[:, -1] = 0
                    data_after_process.append(
                        {
                            "all_tokens": tokens,
                            "position_ids": position_ids,
                            "labels": labels,
                            "ori_seq_len": ori_seq_len,
                            "ori_batch_size": ori_batch_size,
                            "indices": indices,
                            "bin_ids": data_b["bin_ids"],
                            "bin_seqlen": data_b["bin_seqlen"],
                            "pad_size": pad_size,
                            "loss_mask": loss_mask,
                            "old_logprobs": data_b[OLD_TAG],
                            "ref_logprobs": data_b[REF_TAG],
                            "advantages": data_b["advantages"],
                        }
                    )
            else:
                attn_mask, loss_mask, position_ids = generate_loss_mask_position_ids(tokens_, prompt_token_length, response_token_length)
                pad_size = 0
                if self.sp_size > 1:
                    # Pad inputs to ensure seq_len is divisible by sp_size
                    valid_len = tokens_.shape[1]
                    pad_size = math.ceil(valid_len / self.sp_size) * self.sp_size - valid_len

                    tokens = F.pad(tokens_, (0, pad_size), value=self.tokenizer.pad_token_id)
                    position_ids = F.pad(position_ids, (0, pad_size), value=pad_size)

                    labels = torch.roll(tokens_, shifts=-1, dims=1)

                    # Split tensor by sp_size
                    sp_group = get_sp_parallel_group()
                    sp_local_rank = dist.get_rank(sp_group)
                    tokens = sp_split(input_tensor=tokens, split_dim=1, sp_size=self.sp_size, sp_local_rank=sp_local_rank)
                    labels = sp_split(input_tensor=labels, split_dim=1, sp_size=self.sp_size, sp_local_rank=sp_local_rank)
                else:
                    tokens = tokens_
                    labels = torch.roll(tokens, shifts=-1, dims=1)
                if not training:
                    data_after_process.append(
                        {
                            "all_tokens": tokens,
                            "position_ids": position_ids,
                            "ori_seq_len": ori_seq_len,
                            "ori_batch_size": ori_batch_size,
                            "labels": labels,
                            "pad_size": pad_size,
                        }
                    )
                else:
                    loss_mask = torch.roll(loss_mask, shifts=-1, dims=1)
                    loss_mask[:, -1] = 0
                    data_after_process.append(
                        {
                            "all_tokens": tokens,
                            "position_ids": position_ids,
                            "ori_seq_len": ori_seq_len,
                            "ori_batch_size": ori_batch_size,
                            "labels": labels,
                            "pad_size": pad_size,
                            "loss_mask": loss_mask,
                            "old_logprobs": data_b[OLD_TAG],
                            "ref_logprobs": data_b[REF_TAG],
                            "advantages": data_b["advantages"],
                        }
                    )
        return minibatch_size_per_rank, response_token_length_total, data_after_process

    def train_step(self, data_list):
        """
        data_list: list of micro batchs [micro_bs0, micro_bs1]
        """
        self.model.train()  # reset model state
        self.optimizer.zero_grad()
        pg_loss_list = []
        entropy_loss_list = []
        kl_loss_list = []
        micro_bs_num = len(data_list)
        sp_group = get_sp_parallel_group()
        minibatch_size_per_rank, response_token_length_total, data_list = self.preprocess_data_list(data_list=data_list, training=True)
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
            pg_loss_list.append(pg_loss)

            # entropy loss
            entropy_loss = torch.masked_select(entropy, inputs["loss_mask"].bool())
            entropy_loss_mean = torch.sum(entropy_loss) / response_token_length_total * self.fsdp_size
            entropy_loss_list.append(entropy_loss)
            # kl loss
            kl = inputs["ref_logprobs"] - logprobs
            ratio = torch.exp(kl)
            assert not torch.isinf(ratio).any(), "kl loss ratio has inf values"
            assert not torch.isnan(ratio).any(), "kl loss ratio has nan values"
            kld = (ratio - kl - 1).contiguous()
            kl_loss = torch.clamp(kld, min=-10, max=10)
            kl_loss = torch.masked_select(kl_loss, inputs["loss_mask"].bool())
            kl_loss_mean = torch.sum(kl_loss) / response_token_length_total * self.fsdp_size
            kl_loss_list.append(kl_loss)

            # compute backward loss
            pg_loss_mean = torch.sum(pg_loss) / response_token_length_total * self.fsdp_size
            total_loss = pg_loss_mean
            if self.module_args.entropy_coef > 0:
                total_loss = total_loss - self.module_args.entropy_coef * entropy_loss_mean
            if self.module_args.kl_coef > 0:
                total_loss = total_loss + self.module_args.kl_coef * kl_loss_mean
            total_loss.backward()


        # refs to https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#gradient-clipping-and-optimizer-with-dtensor 
        # but results seems not right in torch 2.6.0+cu124
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.module_args.optimizer.clip_grad).detach().item()
        grad_norm = self.fsdp2_clip_grad_norm_(self.model.parameters(), max_norm=self.module_args.optimizer.clip_grad).detach().item()

        self.optimizer.step()
        self.optimizer.zero_grad()

        # collect metric
        pg_loss = (torch.sum(torch.cat(pg_loss_list)) / response_token_length_total * self.fsdp_size).detach().item()
        kl_loss = torch.mean(torch.cat(kl_loss_list)).detach().item()
        entropy_loss = torch.mean(torch.cat(entropy_loss_list)).detach().item()

        train_stats = {
            "pg_loss": pg_loss,
            "kl_loss": kl_loss,
            "entropy_loss": entropy_loss,
            "grad_norm": grad_norm,
        }
        self._metric_list.append(train_stats)

    def forward_step(self, data):
        total_size = data['all_tokens'].shape[0]
        ori_seq_len = data['all_tokens'].shape[1]

        _, _, data_list = self.preprocess_data_list(data_list=[data], training=False)
        # Logprobs holder
        output_logprobs = torch.empty(total_size, ori_seq_len, dtype=torch.bfloat16)
        token_in_seq = []
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
            if self.packing:
                output_logprobs[torch.tensor(inputs["bin_ids"])] = logprobs.cpu()
                token_in_seq.append(sum(inputs["bin_seqlen"]))
            else:
                output_logprobs = logprobs.cpu()
        rank_caculate = torch.distributed.get_rank()
        tag = OLD_TAG
        if OLD_TAG in data.keys():
            tag = REF_TAG
        data.update({tag: output_logprobs})

        return data
