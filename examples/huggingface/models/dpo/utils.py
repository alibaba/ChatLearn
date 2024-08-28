import torch
from typing import Optional

from chatlearn import DeepSpeedModule
import torch.nn.functional as F
from data.reward_dataset import RewardDataset

def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

def concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens, tokenizer):
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """

    def pad_to_length(tensor, length, pad_value, dim=-1):
        if tensor.size(dim) >= length:
            return tensor
        else:
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            return torch.cat(
                [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
            )

    max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
    inputs_ids = torch.cat(
        (
            pad_to_length(chosen_ids, max_length, tokenizer.pad_token_id),
            pad_to_length(reject_ids, max_length, tokenizer.pad_token_id),
        ),
        dim=0,
    )
    max_length = max(c_mask.shape[1], r_mask.shape[1])
    att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
    return inputs_ids, att_masks, prompt_id_lens * 2

def _get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    attention_mask,
    prompt_id_lens,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert average_log_prob == False
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    loss_masks = attention_mask.clone().bool()
    # mask prompts
    for mask, source_len in zip(loss_masks, prompt_id_lens):
        mask[:source_len] = False
    loss_masks = loss_masks[:, 1:]

    # dummy token; we'll ignore the losses on these tokens later
    labels[loss_masks == False] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logprobs_sums = (per_token_logps * loss_masks).sum(-1)
    logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
    return logprobs_sums, logprobs_means

class DPOModel(DeepSpeedModule):

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if return_output:
            return output if num_actions is None else (log_probs[:, -num_actions:], output)
        else:
            return log_probs[:, -num_actions:]
    

    def concatenated_forward(
        self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens = concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens, self.tokenizer
        )
        output = self.forward(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = _get_batch_logps(
            all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=False
        )
        chosen_logps = all_logps_sum[: chosen_ids.shape[0]]
        rejected_logps = all_logps_sum[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: chosen_ids.shape[0]].mean()
