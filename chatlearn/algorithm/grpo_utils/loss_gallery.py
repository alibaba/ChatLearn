"""pg loss"""
import torch

def loss_masked_mean(values, loss_mask, axis=None):
    valid_values = torch.where(loss_mask.bool(), values, 0.0)
    s = (valid_values * loss_mask).sum(axis=axis)
    return s / (loss_mask.sum(axis=axis) + 1e-8)

def calculate_grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    diff_clip_ratio: float = 10,
    pos_clip_ratio: float = 0.2,
    neg_clip_ratio: float = 0.2,
    final_clip_ratio: float = 0.01,
    use_group_sequence_policy: bool = False,
    loss_mask: torch.Tensor = None
):

    logprobs_diff = log_probs - old_log_probs
    # clip logprobs_diff before exp to avoid overflow
    if use_group_sequence_policy:
        logprobs_diff_mean = loss_masked_mean(logprobs_diff.detach(), loss_mask, axis=1)
        seq_logprobs_diff = log_probs - log_probs.detach() + logprobs_diff_mean.unsqueeze(1)
        logprobs_diff = torch.clamp(seq_logprobs_diff, max=diff_clip_ratio)
    else:
        logprobs_diff = torch.clamp(logprobs_diff, max=diff_clip_ratio)

    ratio = torch.exp(logprobs_diff)
    advantages.unsqueeze_(-1)

    pg_loss = -advantages * ratio
    # Upper and lower bound clip
    is_positive_clipped = (ratio > (1 + pos_clip_ratio)) * (advantages > 0)
    is_negative_clipped = (ratio < (1 - neg_clip_ratio)) * (advantages < 0)

    pg_loss_2 = -advantages * torch.clamp(
        ratio, 1 - neg_clip_ratio, 1 + pos_clip_ratio
    )
    pg_loss_clip = torch.max(pg_loss, pg_loss_2)

    is_clipped = pg_loss_2 > pg_loss
    pg_loss_upperbound = torch.ones_like(pg_loss) * final_clip_ratio
    # final clip on loss
    loss = torch.min(pg_loss_clip, pg_loss_upperbound)

    # check pg_loss nan
    assert not torch.isnan(loss).any(), "pg loss is nan"

    return (loss.contiguous(), is_positive_clipped, is_negative_clipped, is_clipped)
