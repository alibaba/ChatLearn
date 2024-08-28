from typing import Tuple
import torch
from chatlearn import DeepSpeedModule
import torch.nn as nn
from .utils import DPOModel
import torch.nn.functional as F

class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards

class PolicyTrainer(DPOModel):

    def setup(self):
        super().setup()
        self.beta = 0.01
        self.label_smoothing = 0
        self.ipo = False
        self.loss_fn = DPOLoss(self.beta, self.label_smoothing, self.ipo)
        self.aux_loss = False
        self.nll_loss = False
        self.aux_loss_coef = 0
        self.nll_loss_coef = 0
        self.acc_mean = 0
        self.loss_mean = 0

    def train_step(self, data_list, iteration):
        self.model.train()  # reset model state
        for data in data_list:
            chosen_ids = data["chosen_input_ids"].squeeze(1).cuda()
            c_mask = data["chosen_attention_mask"].squeeze(1).cuda()
            reject_ids = data["reject_attention_mask"].squeeze(1).cuda()
            r_mask = data["reject_attention_mask"].squeeze(1).cuda()
            reference_chosen_logps = data["reference_chosen_logps"].cuda()
            reference_rejected_logps = data["reference_rejected_logps"].cuda()
            prompt_id_lens = data["extra"]

            chosen_logps, rejected_logps, aux_loss, nll_loss = self.concatenated_forward(
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
            )
            preference_loss, chosen_reward, reject_reward = self.loss_fn(
                chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            if not self.aux_loss:
                aux_loss = 0
            if not self.nll_loss:
                nll_loss = 0

            loss = preference_loss + aux_loss * self.aux_loss_coef + nll_loss * self.nll_loss_coef
            self.model.backward(loss)
            self.model.step()
            acc = (chosen_reward > reject_reward).float().mean().item()
            self.acc_mean = self.acc_mean * 0.9 + 0.1 * acc
            self.loss_mean = self.loss_mean * 0.9 + 0.1 * preference_loss.item()
            # dpo logs
            logs_dict = {
                "loss": preference_loss.item(),
                "acc": acc,
                "chosen_reward": chosen_reward.mean().item(),
                "reject_reward": reject_reward.mean().item(),
                "loss_mean": self.loss_mean,
                "acc_mean": self.acc_mean,
            }
            if self.nll_loss:
                logs_dict["nll_loss"] = nll_loss.item()
            self._logger.info('\t'.join(f"{key}: {value}" for key, value in logs_dict.items()))
