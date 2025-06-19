import gc
import itertools
import logging
import os
import time
from copy import deepcopy
from typing import Any, Dict, List

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tabulate import tabulate
from torch.nn import functional as F
from tqdm import tqdm
from yagm.data.datasets.base_dataset import BaseDataset
from yagm.tasks.base_task import BaseTask
from yagm.transforms.keypoints.decode import decode_heatmap_3d
from yagm.transforms.monai_custom import CustomProbNMS
from yagm.utils import misc as misc_utils
from yagm.utils.misc import get_xlsx_copiable_metrics

from byu.utils.data import SubmissionDataFrame
from byu.utils.metrics import compute_metrics, kaggle_score

logger = logging.getLogger(__name__)


class Heatmap2dTask(BaseTask):
    """
    Task: 2D UNet to predict 2D heatmap
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_epoch_idxs = []
        self.val_pred_results = {}

        self.target_spacing = cfg.data.transform.target_spacing

    def on_validation_start(self):
        logger.info("ON VALIDATION START")
        val_dataset = self.trainer.val_dataloaders.dataset
        self.tta_transforms = val_dataset.tta_transforms
        self.eval_tta_gt_df = val_dataset.tta_gt_df
        val_dataset.fast_val_tomo_loader.start()
        return super().on_validation_start()

    def on_validation_end(self):
        logger.info("ON VALIDATION END")
        val_dataset = self.trainer.val_dataloaders.dataset
        val_dataset.fast_val_tomo_loader.shutdown()
        return super().on_validation_end()

    def on_test_start(self):
        test_dataset = self.trainer.test_dataloaders.dataset
        self.tta_transforms = test_dataset.tta_transforms
        self.eval_tta_gt_df = test_dataset.tta_gt_df
        test_dataset.fast_val_tomo_loader.start()
        return super().on_test_start()

    def on_test_end(self):
        test_dataset = self.trainer.test_dataloaders.dataset
        test_dataset.fast_val_tomo_loader.shutdown()
        return super().on_test_end()

    def forward(self, image):
        return self.model(image)

    def shared_step(self, model, stage, batch, batch_idx, dataloader_idx=None):
        # print('INPUT BATCH:', {k: [v.shape, v.dtype] for k, v in batch.items()})
        batch_img = batch["image"]
        assert batch_img.dtype == torch.uint8
        outs = model(batch_img.half())
        kpt_pred, kptness_pred, heatmap_pred, dsnt_kpt_pred = outs
        if stage == "train" and batch_idx == 0:
            logger.info(
                "First batch:\nInput: %s\nOutput: %s",
                [(k, v.shape) for k, v in batch.items()],
                [e.shape for e in outs],
            )

        if stage == "train":
            batch_heatmap_gt = batch["heatmap_gt"]
            if heatmap_pred.shape[2:] != batch_heatmap_gt.shape[2:]:
                heatmap_pred = F.interpolate(
                    heatmap_pred, size=batch_heatmap_gt.shape[2:], mode="bilinear"
                )
            loss = self.loss_func(
                kpt_pred,
                batch["kpt_gt"],
                batch["kpt_mask"],
                kptness_pred,
                batch["kptness_gt"],
                heatmap_pred,
                batch_heatmap_gt,
                dsnt_kpt_pred,
            )

            # patch so that loss function always resulted in  a dictionary
            if isinstance(loss, dict):
                loss_dict = loss
            elif not isinstance(loss, torch.Tensor):
                loss_dict = {"loss": loss}
            else:
                raise ValueError
        else:
            loss_dict = {"loss": torch.tensor(0.0, dtype=torch.float32)}

        # Lightning: return dictionary with "loss" key
        step_output = {"loss": loss_dict["loss"]}
        step_metrics = {f"{stage}/{k}": v.item() for k, v in loss_dict.items()}
        if stage != "train":
            # the first multiscale prediction mask is used in evaluation
            step_output["heatmap_pred"] = heatmap_pred
        return step_output, step_metrics

    def decode_heatmap(self, pred_heatmap):
        cfg = self.cfg
        radius_voxel = max([1000 / e for e in self.target_spacing])
        # decode
        ret = []
        for channel_idx, heatmap in enumerate(pred_heatmap):
            # print(channel_idx, heatmap.min(), heatmap.max(), heatmap.mean())
            # if channel_idx == 0:
            #     np.save(f'{heatmap.max().item()}.npy', heatmap.cpu().numpy())
            if cfg.task.decode.method == "nms":
                radius_thres = (
                    cfg.task.decode.nms.radius_thres
                    if cfg.task.decode.nms.radius_thres is not None
                    else radius_voxel / 3
                )
                if cfg.task.decode.nms.pool_ksize:
                    pool_ksize = list(cfg.task.decode.nms.pool_ksize)
                else:
                    # @TODO - currently, using fixed pool_ksize=3 due to performance issue
                    # when using large kernel size with Pytorch (>10 sec on 224x448x448)
                    # # maximum ood number which <= radius_thres
                    # # 0.7071067811865475 = 1 / sqrt(2)
                    # pool_ksize = int(
                    #     (2 * radius_thres * 0.7071067811865475 - 1) // 2 * 2 + 1
                    # )
                    # # at least 3, [1,1,1] == no pooling, which increase maxRecall but significantly reduce other metrics
                    # pool_ksize = max(3, pool_ksize)
                    # pool_ksize = [pool_ksize, pool_ksize, pool_ksize]
                    pool_ksize = [3, 3, 3]
                logger.debug(
                    "Heatmap decode with radius=%f, pool_ksize=%s, radius_thres=%s",
                    radius_voxel,
                    pool_ksize,
                    radius_thres,
                )
                outputs = decode_heatmap_3d(
                    heatmap=heatmap,
                    pool_ksize=pool_ksize,
                    nms_radius_thres=radius_thres,
                    blur_operator=None,
                    conf_thres=cfg.task.decode.conf_thres,
                    max_dets=cfg.task.decode.max_dets,
                    timeout=cfg.task.decode.timeout,
                )
                outputs = outputs.cpu()
            else:
                raise ValueError
            ret.append(outputs)
        return ret

    def eval_step_single_model(
        self, model, batch, batch_idx, dataloader_idx=None, stage="val", model_name=None
    ):
        patch_size_in = self.cfg.data.patch_size[1:]  # YX
        heatmap_stride = self.cfg.task.decode.heatmap_stride
        patch_size_out = [round(e / s) for e, s in zip(patch_size_in, heatmap_stride)]
        step_output, step_metrics = self.shared_step(
            model, stage, batch, batch_idx, dataloader_idx=dataloader_idx
        )
        batch_pred_heatmap = step_output["heatmap_pred"]

        if self.cfg.model.decoder.act == "sigmoid":
            batch_pred_heatmap = F.sigmoid(batch_pred_heatmap)
        elif self.cfg.model.decoder.act == "softmax":
            B, C, H, W = batch_pred_heatmap.shape
            batch_pred_heatmap = F.softmax(
                batch_pred_heatmap.view(B, C, H * W), dim=-1
            ).view(B, C, H, W)
        elif (
            self.cfg.model.decoder.act is None
            or self.cfg.model.decoder.act == "identity"
        ):
            pass
        else:
            raise ValueError

        B, C, Y2, X2 = batch_pred_heatmap.shape
        assert C == 1
        # interpolate back to original (finest resolution)
        if [Y2, X2] != patch_size_out:
            logger.debug(
                "PATCH HEATMAP INTERPOLATE: %s --> %s",
                batch_pred_heatmap.shape,
                patch_size_out,
            )
            batch_pred_heatmap = F.interpolate(
                batch_pred_heatmap,
                size=patch_size_out,
                mode="bilinear",
                align_corners=False,
            )
        else:
            logger.debug("PATCH HEATMAP SAME: %s", batch_pred_heatmap.shape)
            pass

        # BCYX -> BC1YX where 1 stand for Z
        batch_pred_heatmap = batch_pred_heatmap[:, :, None]

        model_result = self.val_pred_results.setdefault(
            model_name,
            {
                "cur_pred_heatmap_sum": None,
                "cur_pred_heatmap_count": None,
                "cur_tomo_idx": None,
                "cur_tomo_id": None,
                "cur_tta_idx": None,
                "cur_tta_name": None,
                "cur_target_spacing": None,
                "submission": SubmissionDataFrame(),
            },
        )
        for i in range(B):
            tomo_idx = batch["tomo_idx"][i]
            tta_idx = int(batch["tta_idx"][i].item())
            ori_spacing = batch["ori_spacing"][i].tolist()
            target_spacing = batch["target_spacing"][i].tolist()
            tta_name, tta_transform = self.tta_transforms[tta_idx]
            pred_patch_heatmap = batch_pred_heatmap[i]
            # @TODO - Improve generalization
            # when target heatmap shape Y!=X, rotate 90 cause incompatible heatmap shape
            # easy fix: TTA invert first, then F.interpolate later
            pred_patch_heatmap, weight = tta_transform.invert(pred_patch_heatmap)

            # interpolate to match heatmap_stride

            tomo_shape = batch["tomo_shape"][i].cpu().tolist()
            tomo_heatmap_shape = [
                round(e / s) for e, s in zip(tomo_shape, (1, *heatmap_stride))
            ]
            patch_position = batch["patch_position"][i].cpu()
            is_first = batch["patch_is_first"][i]
            is_last = batch["patch_is_last"][i]
            if stage == "val":
                tomo_id = self.trainer.val_dataloaders.dataset.tomo_ids[tomo_idx]
            elif stage == "test":
                tomo_id = self.trainer.test_dataloaders.dataset.tomo_ids[tomo_idx]
            else:
                raise ValueError

            # print(
            #     f"\ni={i} tomo_idx={tomo_idx} tomo_id={tomo_id} is_first={is_first} is_last={is_last}"
            # )

            if is_first:
                logger.debug(
                    "First patch of tomo %s (index %d) with tta=%s (index %d) received.",
                    tomo_id,
                    tomo_idx,
                    tta_name,
                    tta_idx,
                )
                # @TODO - should handle this case as well
                # assert not is_last
                assert all(
                    [
                        model_result[k] is None
                        for k in [
                            "cur_tomo_idx",
                            "cur_tomo_id",
                            "cur_tta_idx",
                            "cur_tta_name",
                            "cur_target_spacing",
                            "cur_pred_heatmap_sum",
                            "cur_pred_heatmap_count",
                        ]
                    ]
                )

                # allocate new heatmap_sum and heatmap_count
                model_result["cur_tomo_idx"] = tomo_idx
                model_result["cur_tomo_id"] = tomo_id
                model_result["cur_tta_idx"] = tta_idx
                model_result["cur_tta_name"] = tta_name
                model_result["cur_target_spacing"] = target_spacing
                # float16 to save memory
                model_result["cur_pred_heatmap_sum"] = torch.zeros(
                    (C, *tomo_heatmap_shape),
                    dtype=torch.float16,
                    device=batch_pred_heatmap.device,
                )
                model_result["cur_pred_heatmap_count"] = torch.zeros(
                    (C, *tomo_heatmap_shape),
                    dtype=torch.float16,
                    device=batch_pred_heatmap.device,
                )

            assert (
                tomo_idx == model_result["cur_tomo_idx"]
                and tta_name == model_result["cur_tta_name"]
            )
            assert target_spacing == model_result["cur_target_spacing"]
            cur_pred_heatmap_sum = model_result["cur_pred_heatmap_sum"]
            cur_pred_heatmap_count = model_result["cur_pred_heatmap_count"]
            cur_submission: SubmissionDataFrame = model_result["submission"]

            assert (
                cur_pred_heatmap_sum.shape[1:]
                == cur_pred_heatmap_count.shape[1:]
                == tuple(tomo_heatmap_shape)
            ), f"{cur_pred_heatmap_sum.shape[1:]} {cur_pred_heatmap_count.shape[1:]} {tomo_heatmap_shape}"

            roi_start, roi_end, patch_start, patch_end = patch_position.tolist()
            top_pad_z, top_pad_y, top_pad_x = [max(-start, 0) for start in roi_start]
            bot_pad_z, bot_pad_y, bot_pad_x = [
                max(0, end - size) for end, size in zip(roi_end, tomo_shape)
            ]
            assert (
                top_pad_z == bot_pad_z == 0
                and roi_end[0] - roi_start[0] == self.cfg.data.patch_size[0]
            )
            assert (
                tuple(
                    (pe - ps) // stride
                    for pe, ps, stride in zip(
                        patch_end[1:], patch_start[1:], heatmap_stride
                    )
                )
                == pred_patch_heatmap.shape[2:]
            ), f"{tuple((pe - ps) // stride for pe, ps, stride in zip(patch_end[1:], patch_start[1:], heatmap_stride))} {pred_patch_heatmap.shape[2:]}"
            sz, sy, sx = tuple(rs - ps for rs, ps in zip(roi_start, patch_start))
            ez, ey, ex = tuple(
                ps - pe + re
                for ps, pe, re in zip((1, *patch_size_in), patch_end, roi_end)
            )
            assert sz == 0 and ez == 1 and top_pad_z == bot_pad_z == 0
            dst_slices = [
                slice(None),
                slice(
                    round((roi_start[0] + top_pad_z)),
                    round((roi_end[0] - bot_pad_z)),
                ),
                slice(
                    round((roi_start[1] + top_pad_y) / heatmap_stride[0]),
                    round((roi_end[1] - bot_pad_y) / heatmap_stride[0]),
                ),
                slice(
                    round((roi_start[2] + top_pad_x) / heatmap_stride[1]),
                    round((roi_end[2] - bot_pad_x) / heatmap_stride[1]),
                ),
            ]
            src_slices = [
                slice(None),
                slice(
                    round((sz + top_pad_z)),
                    round((ez - bot_pad_z)),
                ),
                slice(
                    round((sy + top_pad_y) / heatmap_stride[0]),
                    round((ey - bot_pad_y) / heatmap_stride[0]),
                ),
                slice(
                    round((sx + top_pad_x) / heatmap_stride[1]),
                    round((ex - bot_pad_x) / heatmap_stride[1]),
                ),
            ]

            cur_pred_heatmap_sum[dst_slices] += pred_patch_heatmap[src_slices] * weight

            cur_pred_heatmap_count[dst_slices] += weight

            if is_last:
                # DECODE HEATMAP TO COORDINATE
                tomo_id = model_result["cur_tomo_id"]
                tta_name = model_result["cur_tta_name"]
                logger.debug(
                    "Last patch of tomo %s (index %d) with tta=%s (index %d) received, decoding..",
                    tomo_id,
                    tomo_idx,
                    tta_name,
                    tta_idx,
                )
                # ensure prediction cover all the tomogram
                # assert torch.all(cur_pred_heatmap_count >= 1.0)
                heatmap = torch.div(
                    cur_pred_heatmap_sum,
                    cur_pred_heatmap_count,
                    out=cur_pred_heatmap_sum,
                )
                outputs = self.decode_heatmap(heatmap)
                assert len(outputs) == 1
                for channel_idx, channel_outputs in enumerate(outputs):
                    assert len(channel_outputs) <= 1
                    if len(channel_outputs) == 1:
                        z, y, x, conf = channel_outputs[0].tolist()
                        # one motor detected
                        cur_submission.add_row(
                            tomo_id=f"{tomo_id}@{tta_name}",
                            x=(x + 0.5)
                            * heatmap_stride[1]
                            * target_spacing[2]
                            / ori_spacing[2]
                            - 0.5,
                            y=(y + 0.5)
                            * heatmap_stride[0]
                            * target_spacing[1]
                            / ori_spacing[1]
                            - 0.5,
                            z=(z + 0.5) * target_spacing[0] / ori_spacing[0] - 0.5,
                            conf=conf,
                        )
                    else:
                        # no motor detected
                        cur_submission.add_row(
                            tomo_id=f"{tomo_id}@{tta_name}",
                            x=-1,
                            y=-1,
                            z=-1,
                            conf=0.0,
                        )

                # clear the cache
                for k in [
                    "cur_tomo_idx",
                    "cur_tomo_id",
                    "cur_tta_idx",
                    "cur_tta_name",
                    "cur_target_spacing",
                    "cur_pred_heatmap_sum",
                    "cur_pred_heatmap_count",
                ]:
                    model_result[k] = None
                gc.collect()
                # torch.cuda.empty_cache()

        # save step output
        # nothing here
        step_save_output = {}
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
        del step_outputs  # just unused

        pred_df = self.val_pred_results[model_name]["submission"].to_pandas()
        del self.val_pred_results[model_name]
        gc.collect()
        gt_df = dataset.tta_gt_df

        ###############################
        # EVALUATE
        metrics = compute_metrics(gt_df, pred_df)
        metrics = {f"{stage}/{k}": v for k, v in metrics.items()}
        metadata = {"pred_df": pred_df}
        return metrics, metadata

    def on_epoch_end_save_metadata(self, metadata, stage, model_name):
        super().on_epoch_end_save_metadata(metadata, stage, model_name)
        save_metadata = {}

        pred_df_save_path = os.path.join(
            self.cfg.env.output_metadata_dir,
            "pred",
            model_name,
            f"ep={self.current_epoch}_step={self.global_step}.csv",
        )
        os.makedirs(os.path.dirname(pred_df_save_path), exist_ok=True)
        metadata["pred_df"].to_csv(pred_df_save_path, index=False)

        # add `stage`/ prefix, e.g val/something
        save_metadata = {f"{stage}/{k}": v for k, v in save_metadata.items()}
        return save_metadata

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
        ret["metrics_trackers"] = self.metrics_tracker
        # grab all saved checkpoints
        best_ckpt_paths = []
        for cb in self.trainer.checkpoint_callbacks:
            best_ckpt_paths.extend(list(cb.best_k_models.keys()))
        best_ckpt_paths = list(set(best_ckpt_paths))

        # list of best checkpoints
        # each checkpoint is identified by its metadata dict
        # with keys: epoch, step
        best_ckpts = []
        if best_ckpt_paths:
            for ckpt_path in best_ckpt_paths:
                ckpt_name = os.path.basename(ckpt_path)
                epoch = int(ckpt_name.split("_")[0].replace("ep=", ""))
                step = int(ckpt_name.split("_")[1].replace("step=", ""))
                best_ckpts.append(
                    {"epoch": epoch, "step": step, "ckpt_path": ckpt_path}
                )
        else:
            # in case of validation/test
            assert not self.cfg.train
            best_ckpts.append({"epoch": 0, "step": 0})
        ret["best_ckpts"] = best_ckpts
        ret["output_metadata_dir"] = os.path.join(self.cfg.env.output_metadata_dir)
        ret["pred_output_dir"] = os.path.join(self.cfg.env.output_metadata_dir, "pred")
        ret["eval_tta_gt_df"] = self.eval_tta_gt_df
        return ret

    def oof_eval(self, all_fold_results, cache=None):
        del cache
        model_names = [e[0] for e in self.val_models]
        logger.info("Available models: %s", model_names)

        oof_eval_tta_gt_dfs = []
        # a dictionary mapping fold_idx to a list
        # each element in the list is a candidate checkpoint metadata
        # we then do a gridsearch over all combination of candidates to compute OOF score
        data = {}  # Dict[int, List[Dict]]
        for fold_idx, fold_results in all_fold_results.items():
            data[fold_idx] = []
            assert fold_idx < self.cfg.cv.num_folds
            best_ckpts = fold_results["best_ckpts"]
            fold_eval_tta_gt_df = fold_results["eval_tta_gt_df"]
            oof_eval_tta_gt_dfs.append(fold_eval_tta_gt_df)
            for ckpt_idx, ckpt_meta in enumerate(best_ckpts):
                epoch, step = ckpt_meta["epoch"], ckpt_meta["step"]
                for model_name in model_names:
                    # the corresponding prediction csv
                    pred_csv_path = os.path.join(
                        fold_results["pred_output_dir"],
                        model_name,
                        f"ep={epoch}_step={step}.csv",
                    )
                    fold_pred_df = pd.read_csv(pred_csv_path)
                    fold_pred_df["fold_idx"] = fold_idx
                    data[fold_idx].append(
                        {
                            "fold_idx": fold_idx,
                            "epoch": epoch,
                            "step": step,
                            "model_name": model_name,
                            "pred_csv_path": pred_csv_path,
                            "pred_df": fold_pred_df,
                            "ckpt_path": ckpt_meta.get("ckpt_path", None),
                        }
                    )
                    logger.debug(
                        "Fold %d [%d/%d] Load fold prediction %s with columns %s from %s",
                        fold_idx,
                        ckpt_idx,
                        len(best_ckpts),
                        fold_pred_df.shape,
                        list(fold_pred_df.columns),
                        pred_csv_path,
                    )

        oof_eval_tta_gt_df = pd.concat(oof_eval_tta_gt_dfs, axis=0).reset_index(
            drop=True
        )
        del oof_eval_tta_gt_dfs, all_fold_results
        gc.collect()

        sorted_fold_idxs = sorted(list(data.keys()))
        entry_idxs = [list(range(len(data[fold_idx]))) for fold_idx in sorted_fold_idxs]
        combines = list(itertools.product(*entry_idxs))

        all_candidate_metrics = []
        for combine in tqdm(combines, desc="OOF gridsearch"):
            assert len(combine) == len(sorted_fold_idxs)
            all_fold_metas = {}
            all_fold_pred_dfs = []
            added_tomo_ids = []
            for fold_idx, candidate_idx in zip(sorted_fold_idxs, combine):
                meta = data[fold_idx][candidate_idx]
                fold_pred_df: pd.DataFrame = meta["pred_df"]
                fold_pred_df = fold_pred_df[
                    ~fold_pred_df["tomo_id"].isin(added_tomo_ids)
                ].reset_index(drop=True)
                fold_tomo_ids = list(fold_pred_df["tomo_id"].unique())
                assert len(set(fold_tomo_ids).intersection(set(added_tomo_ids))) == 0
                added_tomo_ids.extend(fold_tomo_ids)
                all_fold_pred_dfs.append(fold_pred_df)
                all_fold_metas[fold_idx] = {
                    "fold_idx": meta["fold_idx"],
                    "epoch": meta["epoch"],
                    "step": meta["step"],
                    "model_name": meta["model_name"],
                    "ckpt_path": meta["ckpt_path"],
                }
            oof_pred_df = pd.concat(all_fold_pred_dfs, axis=0).reset_index(drop=True)
            metrics = compute_metrics(oof_eval_tta_gt_df, oof_pred_df)
            # kaggle_fbeta = kaggle_score(val_gt_df, oof_pred_df)
            metrics = {f"val/{k}": v for k, v in metrics.items()}
            metrics["__meta__"] = all_fold_metas
            all_candidate_metrics.append(metrics)

        metric_name = self.cfg.task.metric
        metric_mode = self.cfg.task.metric_mode
        all_candidate_metrics.sort(
            key=lambda x: x[metric_name],
            reverse=(metric_mode == "max"),
        )
        best_oof_metrics = all_candidate_metrics[0]
        best_oof_meta = best_oof_metrics.pop("__meta__")

        best_oof_metrics_table = misc_utils.dict_as_table(
            best_oof_metrics,
            sort_by=lambda x: (len(x[0].split("/")), x[0]),
        )
        logger.info(
            "Best %s (mode=%s) is %f, while worst is %s",
            metric_name,
            metric_mode,
            best_oof_metrics[metric_name],
            all_candidate_metrics[-1][metric_name],
        )
        logger.info("BEST OOF METRICS:\n%s", best_oof_metrics_table)
        logger.info("BEST OOF CONFIG:\n%s", best_oof_meta)

        return all_candidate_metrics
