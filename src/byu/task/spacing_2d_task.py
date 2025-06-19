import logging
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from yagm.data.datasets.base_dataset import BaseDataset
from yagm.tasks.base_task import BaseTask
from yagm.utils.misc import get_xlsx_copiable_metrics

from byu.data.datasets.spacing_2d_dataset import MAX_SPACING, MIN_SPACING

logger = logging.getLogger(__name__)


class Spacing2dTask(BaseTask):
    """
    Task: 2D model to predict/regress voxel spacing
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_epoch_idxs = []

    def forward(self, image):
        return self.model(image)

    def shared_step(self, model, stage, batch, batch_idx, dataloader_idx=None):
        # print('INPUT BATCH:', {k: [v.shape, v.dtype] for k, v in batch.items()})
        batch_img = batch["image"]
        target = batch["target"]
        assert batch_img.dtype == torch.uint8

        pred = model(batch_img)
        loss = self.loss_func(pred, target)

        # Lightning: return dictionary with "loss" key
        step_output = {"loss": loss}
        step_metrics = {f"{stage}/loss": loss}
        if stage != "train":
            # the first multiscale prediction mask is used in evaluation
            step_output["pred"] = pred
        return step_output, step_metrics

    def eval_step_single_model(
        self, model, batch, batch_idx, dataloader_idx=None, stage="val", model_name=None
    ):
        step_output, step_metrics = self.shared_step(
            model, stage, batch, batch_idx, dataloader_idx=dataloader_idx
        )
        step_save_output = {
            "idx": batch["idx"].cpu().numpy(),
            "target": batch["target"].cpu().numpy(),
            "pred": step_output["pred"].cpu().numpy(),
        }
        return step_output, step_save_output, step_metrics

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        self.train_epoch_idxs.extend(batch["idx"].cpu().numpy().tolist())
        return super().training_step(batch, batch_idx, dataloader_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return super().validation_step(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError

    def on_train_epoch_end(self):
        logger.info("UNIQUE TRAIN IDXS: %d", len(set(self.train_epoch_idxs)))
        self.train_epoch_idxs.clear()
        return super().on_train_epoch_end()

    def compute_metrics(
        self,
        step_outputs: List[Dict],
        dataset: BaseDataset,
        stage: str = "val",
        model_name=None,
    ):
        gts = np.concatenate([e["target"] for e in step_outputs], axis=0)
        preds = np.concatenate([e["pred"] for e in step_outputs], axis=0)
        assert len(gts) == len(preds) and preds.shape[1] == 1
        preds = preds[:, 0]

        def _nan_fill(preds, value):
            num_nan = np.isnan(preds).any(axis=-1).sum()
            if num_nan > 0:
                logger.warning("Detect %s nan predictions, fill to %f", num_nan, value)
                return np.nan_to_num(preds, nan=value)
            return preds

        preds = _nan_fill(preds, 0.5)
        print(gts.shape, preds.shape)
        diffs = np.abs(preds - gts) * (MAX_SPACING - MIN_SPACING)
        mae = float(diffs.mean())
        min_diff = float(diffs.min())
        max_diff = float(diffs.max())
        std_diff = float(np.std(diffs))
        metrics = {
            f"{stage}/MAE": mae,
            f"{stage}/MIN": min_diff,
            f"{stage}/MAX": max_diff,
            f"{stage}/STD": std_diff,
        }
        metadata = {}
        return metrics, metadata

    def on_epoch_end_save_metadata(self, metadata, stage, model_name):
        super().on_epoch_end_save_metadata(metadata, stage, model_name)
        metadata = {}
        # add `stage`/ prefix, e.g val/something
        metadata = {f"{stage}/{k}": v for k, v in metadata.items()}
        return metadata

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        copiable_metrics_str = get_xlsx_copiable_metrics(
            self.metrics_tracker.all_best_metrics, self.cfg.task.log_metrics
        )
        logger.info(
            "COPIABLE METRICS at epoch=%f step=%d:\n%s",
            self.current_exact_epoch,
            self.global_step,
            copiable_metrics_str,
        )

    def on_test_epoch_end(self) -> None:
        # post processing
        # calculate & log metric
        super().on_test_epoch_end()

    def on_predict_epoch_end(self):
        print("On predict epoch end..")
        raise NotImplementedError

    def get_single_fold_results(self):
        ret = {}
        return ret

    def oof_eval(self, all_fold_results, cache=None):
        raise NotImplementedError
