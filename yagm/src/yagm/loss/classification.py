import logging
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "BiKLDivLoss",
    "WeightedMSELoss",
    "WeightedL1Loss",
    "WeightedSmoothL1Loss",
    "MSEWithLogitsLoss",
    "L1WithLogitsLoss",
    "SmoothL1WithLogitsLoss",
    "BCEWithLogitsLoss",
    "SmoothBCEWithLogitsLoss",
]

logger = logging.getLogger(__name__)

EPS = torch.finfo(torch.float16).eps


class BiKLDivLoss(nn.modules.loss._Loss):

    def __init__(self, reduction="batchmean", weights=[1.0, 0.0]):
        super().__init__()
        self.reduction = reduction
        self.forward_w, self.backward_w = weights

    def forward(self, input, target):
        """Args:
        input: raw logits (before softmax)
        target: probabilities in range[0, 1], sum up to 1 for each sample
        """
        input = F.log_softmax(input, dim=1)
        loss = 0
        if self.forward_w > 0:
            loss += self.forward_w * F.kl_div(
                input, target, reduction=self.reduction, log_target=False
            )
        if self.backward_w > 0:

            target = torch.clamp(target, min=EPS, max=1 - EPS)
            loss += self.backward_w * F.kl_div(
                target.log(), input, reduction=self.reduction, log_target=True
            )
        return loss


class WeightedMSELoss(nn.MSELoss):

    def __init__(self):
        return super().__init__(reduction="none")

    def forward(self, pred, target, weight):
        loss = super().forward(pred, target)
        loss = (loss * weight).sum() / (weight.sum() + EPS)
        return loss


class WeightedL1Loss(nn.L1Loss):

    def __init__(self):
        return super().__init__(reduction="none")

    def forward(self, pred, target, weight):
        loss = super().forward(pred, target)
        loss = (loss * weight).sum() / (weight.sum() + EPS)
        return loss


class WeightedSmoothL1Loss(nn.SmoothL1Loss):

    def __init__(self, beta=1.0):
        return super().__init__(reduction="none", beta=beta)

    def forward(self, pred, target, weight):
        loss = super().forward(pred, target)
        loss = (loss * weight).sum() / (weight.sum() + EPS)
        return loss


class MSEWithLogitsLoss(nn.MSELoss):

    def forward(self, input, target):
        input = F.softmax(input, dim=1)
        return super().forward(input, target)


class L1WithLogitsLoss(nn.L1Loss):

    def forward(self, input, target):
        input = F.softmax(input, dim=1)
        return super().forward(input, target)


class SmoothL1WithLogitsLoss(nn.SmoothL1Loss):

    def forward(self, input, target):
        input = F.softmax(input, dim=1)
        return super().forward(input, target)


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def __init__(self, weight=None, reduction: str = "mean", pos_weight=None) -> None:
        assert reduction in ["none", "mean", "sum", "batch_mean"]
        super().__init__(weight=weight, reduction=reduction, pos_weight=pos_weight)

    def forward(self, input, target):
        if self.reduction != "batch_mean":
            return super().forward(input, target)
        else:
            bs = target.size(0)
            return (
                F.binary_cross_entropy_with_logits(
                    input,
                    target,
                    self.weight,
                    pos_weight=self.pos_weight,
                    reduction="sum",
                )
                / bs
            )


class SmoothBCEWithLogitsLoss(nn.Module):
    """BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """

    def __init__(
        self,
        smoothing=0.1,
        target_threshold: Optional[float] = None,
        weight=None,
        reduction: str = "mean",
        pos_weight=None,
    ):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.target_threshold = target_threshold
        self.reduction = reduction
        if weight is not None:
            weight = torch.Tensor(weight)
        if pos_weight is not None:
            pos_weight = torch.Tensor(pos_weight)
        self.bce = BCEWithLogitsLoss(
            weight=weight, reduction=reduction, pos_weight=pos_weight
        )

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]

        # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
        num_classes = x.shape[-1]
        # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
        if num_classes > 1:
            off_value = self.smoothing / num_classes
            on_value = 1.0 - self.smoothing + off_value
            target = target.long().view(-1, 1)
            target = torch.full(
                (target.size()[0], num_classes),
                off_value,
                device=x.device,
                dtype=x.dtype,
            ).scatter_(1, target, on_value)
        elif num_classes == 1:
            off_value = self.smoothing / 2
            on_value = 1.0 - self.smoothing + off_value
            target = torch.where(target.bool(), on_value, off_value).view(-1, 1)
        else:
            raise AssertionError()

        if self.target_threshold is not None:
            # Make target 0, or 1 if threshold set
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        return self.bce(x, target)
