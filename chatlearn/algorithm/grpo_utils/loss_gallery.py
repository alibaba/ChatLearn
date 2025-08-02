"""pg loss"""
import torch

def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    # If NaNs exist out of mask, replace NaNs in values with a value that
    # won't affect the sum (e.g., 0 for masked regions)
    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(axis=axis)

def masked_mean(values, mask, axis=None):
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)

def calculate_grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    diff_clip_ratio: float = 10,
    pos_clip_ratio: float = 0.2,
    neg_clip_ratio: float = 0.2,
    final_clip_ratio: float = 0.01,
    use_sequence_loss: bool = False,
    loss_mask: torch.Tensor = None
):
    #torch.Size([8, 2048])
    logprobs_diff = log_probs - old_log_probs
    # clip logprobs_diff before exp to avoid overflow
    if use_sequence_loss:
        seq_negative_approx_kl = masked_mean(logprobs_diff.detach(), loss_mask, axis=1)
        negative_approx_kl = log_probs - log_probs.detach() + seq_negative_approx_kl.unsqueeze(1)
        logprobs_diff = torch.clamp(negative_approx_kl, max=diff_clip_ratio)
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

    return (
        loss.contiguous(), 
        is_positive_clipped, 
        is_negative_clipped,
        is_clipped,
    )
