import logging

import torch
from torch import nn
from torch.nn import functional as F
from yagm.loss.mtl import MTLWeightedLoss

from byu.loss.monai_tversky import TverskyLoss

logger = logging.getLogger(__name__)


class NumericalStableBCE(nn.BCEWithLogitsLoss):
    def __init__(self, divisor):
        self.divisor = divisor
        super().__init__(reduction="sum")

    def forward(self, pred, target):
        loss_sum = super().forward(pred, target)
        return loss_sum / self.divisor


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, reduction="mean", pos_weight=None):
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32).reshape(
                1, -1, 1, 1, 1
            )  # BCHWD
        super().__init__(reduction=reduction, pos_weight=pos_weight)


class MtlBCEWithLogitsLoss(nn.Module):

    def __init__(
        self,
        pos_weight=None,
        mtl_method="scalar",
        mtl_loss_weights=None,
        mtl_num_losses=5,
        mtl_names=None,
        mtl_si=False,
    ):
        super().__init__()
        self.mtl_num_losses = mtl_num_losses
        if pos_weight:
            assert len(pos_weight) == mtl_num_losses
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32).reshape(
                1, -1, 1, 1, 1
            )  # BCHWD
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

        if mtl_loss_weights is None:
            mtl_loss_weights = [1.0] * mtl_num_losses

        if mtl_method == "scalar":
            wsum = sum(mtl_loss_weights)
            mtl_loss_weights = [e / wsum for e in mtl_loss_weights]
            logger.info(
                "MTL method is `scalar`, normalize loss weights to %s", mtl_loss_weights
            )
        self.mtl = MTLWeightedLoss(
            weight_method=mtl_method,
            loss_weights=mtl_loss_weights,
            loss_names=mtl_names,
            task_coefs=(1.0, 1.0, 1.0, 1.0, 1.0),
            scale_invariant=mtl_si,
        )

    def get_coefs(self):
        return self.mtl.get_coefs()

    def set_epoch(self, epoch):
        return self.mtl.set_epoch(epoch)

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: (B, C, *)
            target: (B, C, *)
        """
        vol_loss = self.bce(pred, target)
        assert vol_loss.size(1) == self.mtl_num_losses
        chan_losses = [vol_loss[:, i].mean() for i in range(vol_loss.shape[1])]
        loss, _ = self.mtl(*chan_losses)
        return loss


# for testing purpose only
# just equal to multiply loss by `multiplier`**2
class _SimpleMSE(nn.MSELoss):
    def __init__(self, multiplier):
        self.multiplier = multiplier
        super().__init__(reduction="mean")

    def forward(self, input, target):
        input = input * self.multiplier
        target = target * self.multiplier
        return super().forward(input, target)


class ForegroundWeightedHeatmapLoss(nn.Module):
    def __init__(self, loss_name, fg_weight=1.0, fg_thres=0.0):
        super().__init__()
        assert fg_weight >= 1.0
        self.loss_name = loss_name
        self.fg_weight = fg_weight
        self.fg_thres = fg_thres
        if loss_name == "mse":
            loss_func = nn.MSELoss(reduction="none")
        elif loss_name == "l1":
            loss_func = nn.L1Loss(reduction="none")
        elif loss_name == "bce":
            loss_func = nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise ValueError
        self.loss_func = loss_func

    def forward(self, pred, target):
        loss = self.loss_func(pred, target)
        if self.fg_weight != 1.0:
            weight = (target > self.fg_thres) * (self.fg_weight - 1.0) + 1.0
            loss = loss * weight
        loss = torch.mean(loss)
        return loss


class SoftmaxHeatmapLoss(nn.Module):
    def __init__(self, loss_name):
        super().__init__()
        if loss_name == "cce":
            loss_func = nn.CrossEntropyLoss(
                ignore_index=-100, reduction="mean", label_smoothing=0.0
            )
        elif loss_name.startswith("tversky_"):
            coef = float(loss_name.replace("tversky_", "", 1))
            assert coef > 0
            alpha = 1 / (1 + coef**2)
            beta = 1 - alpha
            loss_func = TverskyLoss(
                include_background=True,
                to_onehot_y=False,
                sigmoid=False,
                softmax=True,
                other_act=None,
                alpha=alpha,
                beta=beta,
                reduction="mean",
                smooth_nr=1e-2,
                smooth_dr=1e-2,
                batch=False,
                soft_label=True,
            )
        else:
            raise ValueError
        self.loss_func = loss_func

    def prepair_softmax_target(self, fg_target):
        """N-channels foreground heatmap to N+1 (include background) channels.
        This only works if positive pixels in each foreground channels are not overlap.
        """
        # target has shape (B, C, ...)
        bg_target = 1.0 - torch.sum(fg_target, dim=1, keepdim=True)  # shape (B, 1, ...)
        target = torch.cat([bg_target, fg_target], dim=1)  # shape (B, C+1,...)
        return target

    def forward(self, pred, target):
        target = self.prepair_softmax_target(target)
        loss = self.loss_func(pred, target)
        return loss


class MultiscaleSegmentationLossWrapper(nn.Module):
    def __init__(self, loss_func, num_losses=1, mtl_cfg=None):
        super().__init__()
        self.loss_func = loss_func
        self.mtl = MTLWeightedLoss(
            weight_method=getattr(mtl_cfg, "method", "scalar"),
            loss_weights=getattr(mtl_cfg, "weights", [1.0] * num_losses),
            loss_names=getattr(mtl_cfg, "names", None),
            task_coefs=getattr(mtl_cfg, "task_coefs", [1.0] * num_losses),
            scale_invariant=getattr(mtl_cfg, "si", False),
        )

    def get_coefs(self):
        if hasattr(self.mtl, "get_coefs"):
            return self.mtl.get_coefs()
        else:
            return None

    def set_epoch(self, epoch):
        if hasattr(self.mtl, "set_epoch"):
            self.mtl.set_epoch(epoch)

    def forward(self, ms_preds, target):
        ret = {}
        ms_losses = []
        for scale_idx, cur_scale_pred in enumerate(ms_preds):
            if cur_scale_pred.shape != target.shape:
                logger.debug(
                    "LOSS INTERPOLATE: %s --> %s", target.shape, cur_scale_pred.shape
                )
                cur_scale_target = F.interpolate(
                    target,
                    cur_scale_pred.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            else:
                cur_scale_target = target
                logger.debug("LOSS SAME: %s", target.shape)

            loss = self.loss_func(cur_scale_pred, cur_scale_target)
            if isinstance(loss, dict):
                # dictionary of different loss components, e.g BCE + Tversky combo
                cur_scale_loss = loss["loss"]
                for k, v in loss.items():
                    ret[f"{k}_{scale_idx}"] = v
            else:
                # tensor with 1 element
                cur_scale_loss = loss
                ret[f"loss_{scale_idx}"] = loss
            ms_losses.append(cur_scale_loss)
        loss, _ = self.mtl(*ms_losses)
        ret["loss"] = loss
        return ret


class CombineLoss(nn.Module):

    def __init__(self, losses, pre_scale, combine_mtl):
        super().__init__()
        num_losses = len(losses)
        losses_dict = {}
        self.pre_scales = []
        for single_loss in losses:
            assert len(single_loss) == 1
            for loss_name, loss_func in single_loss.items():
                assert loss_name not in losses_dict
                losses_dict[loss_name] = loss_func
                self.pre_scales.append(pre_scale[loss_name])
        self.losses = nn.ModuleDict(losses_dict)
        self.combine_mtl = MTLWeightedLoss(
            weight_method=combine_mtl.get("method", "scalar"),
            loss_weights=combine_mtl.get("weights", [1.0] * num_losses),
            loss_names=list(losses_dict.keys()),
            task_coefs=combine_mtl.get("task_coefs", [1.0] * num_losses),
            scale_invariant=combine_mtl.get("si", False),
        )

    def forward(self, pred, target):
        ret = {}
        losses = []
        for loss_idx, (loss_name, loss_func) in enumerate(self.losses.items()):
            loss = loss_func(pred, target) * self.pre_scales[loss_idx]
            losses.append(loss)
            ret[f"loss_{loss_name}"] = loss
        loss, _ = self.combine_mtl(*losses)
        ret["loss"] = loss
        return ret
