# pylint: skip-file
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from chatlearn import FSDPModule
from chatlearn.utils import to_device
from chatlearn.utils.communication_op import gather, get_sp_parallel_group
from .loss_gallery import calculate_grpo_loss

REF_TAG = "ref_logprobs"
OLD_TAG = "old_logprobs"


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def sp_split(input_tensor, split_dim, sp_size, sp_local_rank):
    return torch.tensor_split(input_tensor, sp_size, split_dim)[sp_local_rank]


def generate_loss_mask_position_ids(
    tokens: torch.Tensor, prompt_token_length: list, response_token_length: list
):
    # Setup attention mask by prompt token length and response token length
    loss_mask = torch.zeros_like(tokens, dtype=torch.int32, device=tokens.device)
    for i, _ in enumerate(prompt_token_length):
        loss_mask[
            i,
            prompt_token_length[i] : prompt_token_length[i] + response_token_length[i],
        ] = 1.0
    _, seq_len = tokens.size()
    position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0).expand_as(tokens)

    return loss_mask, position_ids


class PolicyTrainer(FSDPModule):
    """policy trainer"""

    def setup(self):
        super().setup()
        self._metric_prefix = "policy_trainer"

    def preprocess_data_forward_only(self, data_b):
        tokens_ = data_b["all_tokens"].long()
        prompt_token_length = data_b["prompt_token_length"]
        response_token_length = data_b["response_token_length"]
        _, ori_seq_len = tokens_.size()

        # Pad tokens_ to ensure max valid token length meets sp requirements
        if self.sp_size > 1:
            # Since valid token_length will be 1 less than full token length.
            # We need to make sure valid token can be divided by sp_size.
            # When full_token_len = 2048, sp_size=2, the max valid token will be 2047 which is not divided by sp_size.
            # We need pad original token to full_token_len=2049
            # Then max valid token length will be 2048 which can be divided by sp_size.
            valid_len = tokens_.shape[1] - 1
            pad_size = math.ceil(valid_len / self.sp_size) * self.sp_size - valid_len
            tokens_ = F.pad(tokens_, (0, pad_size), value=self.tokenizer.pad_token_id)

        # Find max seq_len in batch
        max_len = max(
            [
                prompt_len + response_len
                for prompt_len, response_len in zip(
                    prompt_token_length, response_token_length
                )
            ]
        )
        max_valid_len = max_len - 1

        # Modify max_len to be divisible by sp_size
        if self.sp_size > 1:
            max_valid_len = math.ceil(max_valid_len / self.sp_size) * self.sp_size

        # Generate loss mask, positon ids
        _, position_ids = generate_loss_mask_position_ids(
            tokens_, prompt_token_length, response_token_length
        )

        # Cut to max_valid_len in batch
        tokens = tokens_[:, :max_valid_len]
        labels = tokens_[:, 1 : max_valid_len + 1]
        position_ids = position_ids[:, :max_valid_len]

        # Split all inputs required to calculate logprobs on seq_len dim when sp_size > 1
        if self.sp_size > 1:
            sp_group = get_sp_parallel_group()
            sp_local_rank = dist.get_rank(sp_group)
            tokens = sp_split(
                input_tensor=tokens,
                split_dim=1,
                sp_size=self.sp_size,
                sp_local_rank=sp_local_rank,
            )
            labels = sp_split(
                input_tensor=labels,
                split_dim=1,
                sp_size=self.sp_size,
                sp_local_rank=sp_local_rank,
            )

        inputs = {
            "all_tokens": tokens,
            "position_ids": position_ids,
            "labels": labels,
            "ori_seq_len": ori_seq_len,
        }
        for k, v in inputs.items():
            inputs[k] = to_device(torch.cuda.current_device(), v)
        return inputs

    def preprocess_data_train(self, data_b):
        """
        Data preprocess for training
        data_b: dict of tensors
        """

        tokens_ = data_b["all_tokens"].long()
        prompt_token_length = data_b["prompt_token_length"]
        response_token_length = data_b["response_token_length"]
        old_logprobs = data_b[OLD_TAG]
        ref_logprobs = data_b[REF_TAG]
        _, ori_seq_len = tokens_.size()

        # Pad tokens_ to ensure max valid token length meets sp requirements
        if self.sp_size > 1:
            # Since valid token_length will be 1 less than full token length.
            # We need to make sure valid token length can be divided by sp_size.
            # When full_token_len = 2048, sp_size=2, the max valid token will be 2047 which is not divided by sp_size.
            # We need pad original token to full_token_len=2049
            # Then max valid token length will be 2048 which can be divided by sp_size.
            valid_len = tokens_.shape[1] - 1
            pad_size = math.ceil(valid_len / self.sp_size) * self.sp_size - valid_len
            tokens_ = F.pad(tokens_, (0, pad_size), value=self.tokenizer.pad_token_id)
            old_logprobs = F.pad(old_logprobs, (0, pad_size), value=0)
            ref_logprobs = F.pad(ref_logprobs, (0, pad_size), value=0)

        # Find max valid token length in batch
        max_len = max(
            [
                prompt_len + response_len
                for prompt_len, response_len in zip(
                    prompt_token_length, response_token_length
                )
            ]
        )
        max_valid_len = max_len - 1

        # Modify max_len to be divisible by sp_size
        if self.sp_size > 1:
            max_valid_len = math.ceil(max_valid_len / self.sp_size) * self.sp_size

        # Generate loss mask, positon ids
        loss_mask, position_ids = generate_loss_mask_position_ids(
            tokens_, prompt_token_length, response_token_length
        )

        # Cut to max_valid_len in batch
        tokens = tokens_[:, :max_valid_len].contiguous()
        labels = tokens_[:, 1 : max_valid_len + 1].contiguous()
        loss_mask = loss_mask[:, 1 : max_valid_len + 1]
        position_ids = position_ids[:, :max_valid_len]
        old_logprobs = old_logprobs[:, :max_valid_len]
        ref_logprobs = ref_logprobs[:, :max_valid_len]

        # Split all inputs required to calculate logprobs on seq_len dim when sp_size > 1
        if self.sp_size > 1:
            sp_group = get_sp_parallel_group()
            sp_local_rank = dist.get_rank(sp_group)
            tokens = sp_split(
                input_tensor=tokens,
                split_dim=1,
                sp_size=self.sp_size,
                sp_local_rank=sp_local_rank,
            )
            labels = sp_split(
                input_tensor=labels,
                split_dim=1,
                sp_size=self.sp_size,
                sp_local_rank=sp_local_rank,
            )

        inputs = {
            "all_tokens": tokens,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "advantages": data_b["advantages"],
            "prompt_token_length": prompt_token_length,
            "labels": labels,
            "old_logprobs": old_logprobs,
            "ref_logprobs": ref_logprobs,
        }

        for k, v in inputs.items():
            inputs[k] = to_device(torch.cuda.current_device(), v)

        return inputs

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
        for data_b in data_list:
            inputs = self.preprocess_data_train(data_b)
            output = self.model(
                input_ids=inputs["all_tokens"],
                attention_mask=None,
                position_ids=inputs["position_ids"],
                use_cache=False,
            )
            logprobs = logprobs_from_logits(output.logits, inputs["labels"])
            if sp_group is not None:
                logprobs = gather(
                    input_tensor=logprobs, sp_group=sp_group, gather_dim=1
                )
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
            # Reference: https://github.com/pytorch/pytorch/blob/ \
            # c45515c2eda19b1a1ff5762f1571c6fe63773c8a/torch/distributed/fsdp/_runtime_utils.py#L848
            # Since grad will be divided by fsdp world size in backward hook
            # We need to multiple pg_loss_mean by sp_size to avoid mean calculate of grad within dp rank
            pg_loss_mean = torch.mean(pg_loss) / micro_bs_num * self.sp_size
            pg_loss_mean.backward()
            pg_loss_list.append(pg_loss)

            # kl loss
            kl = inputs["ref_logprobs"] - logprobs
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
        grad_norm = (
            self.model.clip_grad_norm_(max_norm=self.module_args.optimizer.clip_grad)
            .detach()
            .item()
        )
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
            "grad_norm": grad_norm,
        }
        self._metric_list.append(train_stats)

    def forward_step(self, data):
        inputs = self.preprocess_data_forward_only(data)
        with torch.no_grad():
            output = self.model(
                input_ids=inputs["all_tokens"],
                attention_mask=None,
                position_ids=inputs["position_ids"],
                use_cache=False,
            )
            sp_group = get_sp_parallel_group()
            logprobs = logprobs_from_logits(output.logits, inputs["labels"])
            if sp_group is not None:
                logprobs = gather(
                    input_tensor=logprobs, sp_group=sp_group, gather_dim=1
                )
            # Repad logprobs to max_seq_len to allow concatenation
            logprobs_len = logprobs.shape[1]
            logprobs = F.pad(
                logprobs,
                (0, inputs["ori_seq_len"] - logprobs_len),
                mode="constant",
                value=0,
            )
        tag = OLD_TAG
        if OLD_TAG in data.keys():
            tag = REF_TAG
        data.update({tag: logprobs})
        return data
