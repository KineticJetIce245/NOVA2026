import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 2.0
ALPHA = None


class FocalLoss(nn.Module):
    def __init__(self, gamma=GAMMA, alpha=ALPHA, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=logits.device)[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
