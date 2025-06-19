import logging

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "ChannelMaskedBiKLDivLoss2d",
    "ChannelMaskedJSDLoss2d",
    "ChannelMaskedBCEWithLogitsLoss2d",
]

logger = logging.getLogger(__name__)


class ChannelMaskedBiKLDivLoss2d(nn.modules.loss._Loss):

    def __init__(self, reduction="batchmean", weights=[1.0, 0.0]):
        super().__init__()
        self.reduction = reduction
        self.forward_w, self.backward_w = weights

    def forward(self, input, target, mask):
        """Args:
        input: raw logits (before softmax) of shape (B, C, H, W)
        target: target heatmap of shape (B, C, H, W)
        mask: (B, C) where True -> keep, False -> ignore
        """
        input = torch.masked_fill(input, ~mask[..., None, None], 1.0)
        target = torch.masked_fill(target, ~mask[..., None, None], 1.0)
        input = torch.flatten(input, -2, -1)
        target = torch.flatten(target, -2, -1)
        # normalize target to a probability distribution
        target = target / target.sum(dim=-1, keepdim=True)
        input = F.log_softmax(input, dim=-1)
        loss = 0
        kl_reduction = "sum" if self.reduction == "mask_sum" else self.reduction
        if self.forward_w > 0:
            loss += self.forward_w * F.kl_div(
                input, target, reduction=kl_reduction, log_target=False
            )
        if self.backward_w > 0:
            epsilon = 1e-15
            target = torch.clamp(target, min=epsilon, max=1 - epsilon)
            loss += self.backward_w * F.kl_div(
                target.log(), input, reduction=kl_reduction, log_target=True
            )
        if self.reduction == "mask_sum":
            mask_sum = mask.sum()
            # prevent divide by 0
            if mask_sum > 0:
                return loss / mask_sum
            else:
                # expect to be 0 as well
                return loss
        else:
            return loss


class ChannelMaskedJSDLoss2d(nn.modules.loss._Loss):
    """Jensen Shannon Divergence Loss for 2D heatmap (B, C, H, W)"""

    def __init__(self, reduction="batchmean"):
        super().__init__()
        self.reduction = reduction
        logger.info("MaskedJSDLoss2d with reduction=%s", reduction)
        self.kl = nn.KLDivLoss(
            reduction="sum" if reduction == "mask_sum" else reduction, log_target=True
        )

    def forward(self, input, target, mask):
        """Args:
        input: raw logits (before softmax) of shape (B, C, H, W)
        target: target heatmap of shape (B, C, H, W)
        mask: (B, C) where True -> keep, False -> ignore
        """
        input = torch.masked_fill(input, ~mask[..., None, None], 1.0)
        target = torch.masked_fill(target, ~mask[..., None, None], 1.0)
        input = torch.flatten(input, -2, -1)
        target = torch.flatten(target, -2, -1)

        input = F.log_softmax(input, dim=-1)

        # normalize target to a probability distribution
        target = target / target.sum(dim=-1, keepdim=True)
        epsilon = 1e-15
        target = torch.clamp(target, min=epsilon, max=1 - epsilon)
        M = ((input.exp() + target) * 0.5).log()
        target = target.log()
        jsd_loss = 0.5 * (self.kl(M, input) + self.kl(M, target))
        if self.reduction != "mask_sum":
            return jsd_loss
        else:
            mask_sum = mask.sum()
            # prevent divide by 0
            if mask_sum > 0:
                return jsd_loss / mask_sum
            else:
                # expect to be 0 as well
                return jsd_loss


class ChannelMaskedBCEWithLogitsLoss2d(nn.Module):

    def __init__(self, weight=None, pos_weight=None) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            weight=weight, reduction="none", pos_weight=pos_weight
        )

    def forward(self, input, target, mask):
        """
        Args:
            input: (B, C, H, W)
            target: (B, C, H, W)
            mask: (B, C), True -> keep, False -> ignore
        """
        loss = self.bce(input, target)
        loss = loss * mask[..., None, None]
        loss = loss.sum()
        mask_sum = mask.sum()
        if mask_sum > 0:
            return loss / mask_sum
        else:
            return loss
