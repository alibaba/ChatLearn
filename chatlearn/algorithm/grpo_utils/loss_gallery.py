"""pg loss"""
import torch


def calculate_grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    diff_clip_ratio: float = 10,
    pos_clip_ratio: float = 0.2,
    neg_clip_ratio: float = 0.2,
    final_clip_ratio: float = 0.01,
):
    logprobs_diff = log_probs - old_log_probs
    # clip logprobs_diff before exp to avoid overflow
    logprobs_diff = torch.clamp(logprobs_diff, max=diff_clip_ratio)

    ratio = torch.exp(logprobs_diff)
    pg_loss = -advantages.unsqueeze(-1) * ratio
    # Upper and lower bound clip
    pg_loss_2 = -advantages.unsqueeze(-1) * torch.clamp(
        ratio, 1 - neg_clip_ratio, 1 + pos_clip_ratio
    )
    pg_loss_clip = torch.max(pg_loss, pg_loss_2)
    pg_loss_upperbound = torch.ones_like(pg_loss) * final_clip_ratio
    # final clip on loss
    loss = torch.min(pg_loss_clip, pg_loss_upperbound)

    # check pg_loss nan
    assert not torch.isnan(loss).any(), "pg loss is nan"

    return loss.contiguous()
