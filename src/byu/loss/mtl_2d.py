from torch import nn
from yagm.loss.classification import (
    WeightedL1Loss,
    WeightedMSELoss,
    WeightedSmoothL1Loss,
)
from yagm.loss.heatmap_2d import (
    ChannelMaskedBCEWithLogitsLoss2d,
    ChannelMaskedBiKLDivLoss2d,
    ChannelMaskedJSDLoss2d,
)
from yagm.loss.mtl import MTLWeightedLoss


class Mtl2dLoss(nn.Module):

    LOSS_NAMES = ["kpt", "kptness", "heatmap", "dsnt"]

    def __init__(
        self,
        kpt_loss="mse",
        kptness_loss="bce",
        heatmap_loss="bce",
        dsnt_loss="mse",
        enable_idxs=(0, 1, 2, 3),
        combine_mtl=None,
        **kwargs
    ):
        super().__init__()
        assert len(enable_idxs) == len(set(enable_idxs)) and len(enable_idxs) <= len(
            self.LOSS_NAMES
        )
        self.enable_idxs = enable_idxs
        self.num_losses = len(enable_idxs)

        if kpt_loss == "mse":
            self.kpt_loss_fn = WeightedMSELoss()
        elif kpt_loss == "l1":
            self.kpt_loss_fn = WeightedL1Loss()
        elif kpt_loss == "smooth_l1":
            self.kpt_loss_fn = WeightedSmoothL1Loss()
        else:
            raise ValueError

        assert kptness_loss == "bce"
        self.kptness_loss_fn = nn.BCEWithLogitsLoss(
            weight=None, reduction="mean", pos_weight=None
        )

        self._heatmap_mask = False
        if heatmap_loss == "cce":
            raise NotImplementedError
            # self.heatmap_loss = smp.losses.SoftCrossEntropyLoss(
            #     reduction='mean', smooth_factor=None, ignore_index=-100, dim=1)
        elif heatmap_loss == "bce":
            self.heatmap_loss_fn = nn.BCEWithLogitsLoss(
                weight=None, reduction="mean", pos_weight=None
            )
        elif heatmap_loss == "mse":
            self.heatmap_loss_fn = nn.MSELoss(reduction="mean")
        elif heatmap_loss == "l1":
            self.heatmap_loss_fn = nn.L1Loss(reduction="mean")
        elif heatmap_loss == "kl":
            self._heatmap_mask = True
            self.heatmap_loss_fn = ChannelMaskedBiKLDivLoss2d(
                reduction=kwargs.get("kl_reduction", "batchmean"), weights=[1.0, 0.0]
            )
        elif heatmap_loss == "bikl":
            self._heatmap_mask = True
            self.heatmap_loss_fn = ChannelMaskedBiKLDivLoss2d(
                reduction=kwargs.get("kl_reduction", "batchmean"), weights=[0.5, 0.5]
            )
        elif heatmap_loss == "jsd":
            self._heatmap_mask = True
            self.heatmap_loss_fn = ChannelMaskedJSDLoss2d(
                reduction=kwargs.get("kl_reduction", "batchmean")
            )
        else:
            raise ValueError

        if dsnt_loss == "mse":
            self.dsnt_loss_fn = WeightedMSELoss()
        elif dsnt_loss == "l1":
            self.dsnt_loss_fn = WeightedL1Loss()

        self.combine_mtl = MTLWeightedLoss(
            weight_method=combine_mtl.get("method", "scalar"),
            loss_weights=combine_mtl.get("weights", [1.0] * self.num_losses),
            loss_names=[self.LOSS_NAMES[idx] for idx in enable_idxs],
            task_coefs=combine_mtl.get("task_coefs", [1.0] * self.num_losses),
            scale_invariant=combine_mtl.get("si", False),
        )

    def set_epoch(self, epoch):
        if hasattr(self.combine_mtl, "set_epoch"):
            self.combine_mtl.set_epoch(epoch)

    def forward(
        self,
        kpt_pred,
        kpt_target,
        kpt_mask,
        kptness_pred,
        kptness_target,
        heatmap_pred,
        heatmap_target,
        dsnt_kpt_pred,
    ):
        ret = {}
        if 0 in self.enable_idxs:
            kpt_loss = self.kpt_loss_fn(kpt_pred, kpt_target, kpt_mask)
            ret["kpt_loss"] = kpt_loss
        if 1 in self.enable_idxs:
            kptness_loss = self.kptness_loss_fn(kptness_pred, kptness_target)
            ret["kptness_loss"] = kptness_loss
        if 2 in self.enable_idxs:
            if not self._heatmap_mask:
                heatmap_loss = self.heatmap_loss_fn(heatmap_pred, heatmap_target)
            else:
                heatmap_loss = self.heatmap_loss_fn(
                    heatmap_pred, heatmap_target, kpt_mask[..., 0].bool()
                )
            ret["heatmap_loss"] = heatmap_loss
        if 3 in self.enable_idxs:
            dsnt_loss = self.dsnt_loss_fn(dsnt_kpt_pred, kpt_target, kpt_mask)
            ret["dsnt_loss"] = dsnt_loss

        losses = list(ret.values())
        total_loss, _ = self.combine_mtl(*losses)
        ret["loss"] = total_loss
        return ret
