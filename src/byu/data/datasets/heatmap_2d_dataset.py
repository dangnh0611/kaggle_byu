import gc
import json
import logging
import os
import random
import time
from functools import partial
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import hydra
import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf
from scipy.stats import truncnorm
from torch.nn import functional as F
from yagm.data.datasets.base_dataset import BaseDataset
from yagm.transforms import albumentations_custom as AC
from yagm.transforms.keypoints.encode import (
    generate_2d_gaussian_heatmap,
    generate_2d_point_mask,
    z_slice_normal_dist_3d,
)
from yagm.transforms.keypoints.helper import kpts_pad_for_batching, kpts_spatial_crop
from yagm.transforms.sliding_window import get_sliding_patch_positions
from yagm.transforms.tta_3d import build_tta

from byu.data.fast_val_loader import FastValTomoLoader
from byu.data.io import WRONG_QUANTILE_TOMO_IDS
from byu.utils.misc import compute_target_spacing_shape

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

logger = logging.getLogger(__name__)


CV2_INTERPOLATION_METHODS = {
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "bilinear": cv2.INTER_LINEAR,
    "trilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def _byu_get_safe_bbox(params, data, margin_xy=(100, 100)):
    """Get safe bbox to crop which contain at least 1 motor.
    Used for a custom Albumentations transform `CustomRandomSizedBBoxSafeCrop`
    """
    # img_h, img_w = params['shape'][:2]
    if "keypoints" in data and len(data["keypoints"]) > 0:
        kpt = random.choice(data["keypoints"])
        # x, y, angle, scale
        assert len(kpt) == 9
        x, y = kpt[:2]
        scale = kpt[3]
        return [
            x - margin_xy[0] * scale,
            y - margin_xy[1] * scale,
            x + margin_xy[0] * scale,
            y + margin_xy[1] * scale,
        ]
    else:
        return None


class Heatmap2dDataset(BaseDataset):
    """
    Dataset for 2D Heatmap Estimation approach,
    typically used in combination with a 2D UNet model
    """

    def __init__(self, cfg, stage="train", cache={}):
        super().__init__(cfg, stage, cache)
        ########
        # UNCOMMENT THIS IF YOU FACE SEGMENT FAULT ISSUE WITHIN PYTORCH DATALOADER
        # by default, Pytorch use multiple threads to speedup CPU operation
        # if not set number of threads to 1 before `fork` of new processes,
        # created by Dataloader with num_workers > 0
        # then boom! segmentfault, related to: https://github.com/pytorch/pytorch/issues/54752
        # each dataloader process then can change the number of threads by a call to torch.set_num_threads()
        # e.g, in worker_init_fn passed to DataLoader
        ########
        # logger.info(
        #     "Temporarily set Pytorch number of threads from %d to 1",
        #     torch.get_num_threads(),
        # )
        # torch.set_num_threads(1)

        self.data_dir = cfg.env.data_dir
        self.target_spacing = cfg.data.transform.target_spacing
        self.sigma = cfg.data.sigma
        # BUILD TRANSFORM FUNC
        if stage == "train":
            self.transform = self.build_train_transform()

        # LOAD LABELS
        gt_df = (
            pl.scan_csv(
                os.path.join(
                    self.data_dir,
                    "processed",
                    f"{cfg.data.label_fname}.csv" if stage == "train" else "gt_v3.csv",
                )
            )
            .sort("tomo_id")
            .with_columns(
                pl.col("motor_zyx").map_elements(eval, return_dtype=pl.Object)
            )
            .collect()
        )
        all_tomo_ids = sorted(gt_df["tomo_id"].unique().to_list())

        # CV Splitting
        with open(
            os.path.join(
                self.data_dir, "processed", "cv", "v3", f"{cfg.cv.strategy}.json"
            ),
            "r",
        ) as f:
            cv_meta = json.load(f)
        assert cv_meta["num_folds"] == cfg.cv.num_folds
        all_val_tomo_ids = []
        for _fold_meta in cv_meta["folds"]:
            all_val_tomo_ids.extend(_fold_meta["val"])
        val_tomo_ids = cv_meta["folds"][cfg.cv.fold_idx]["val"]
        assert len(val_tomo_ids) == len(set(val_tomo_ids))
        train_tomo_ids = sorted(list(set(all_tomo_ids).difference(set(val_tomo_ids))))
        logger.info(
            "CV fold %d (total %d folds) with strategy=%s\nTrain on %d tomos, validate on %d tomos",
            cfg.cv.fold_idx,
            cfg.cv.num_folds,
            cfg.cv.strategy,
            len(train_tomo_ids),
            len(val_tomo_ids),
        )
        if stage == "train":
            if cfg.cv.train_on_all:
                self.tomo_ids = train_tomo_ids + val_tomo_ids
            else:
                self.tomo_ids = train_tomo_ids
        else:
            if cfg.cv.train_on_all:
                self.tomo_ids = all_val_tomo_ids
            else:
                self.tomo_ids = val_tomo_ids

        gt_df = gt_df.filter(pl.col("tomo_id").is_in(self.tomo_ids)).sort("tomo_id")

        # additional filtering
        if stage == "train":
            if cfg.data.filter_rule == "eq1":
                gt_df = gt_df.filter(pl.col("num_motors") == 1)
            elif cfg.data.filter_rule == "le1":
                gt_df = gt_df.filter(pl.col("num_motors") <= 1)
            elif cfg.data.filter_rule is None:
                pass
            else:
                raise ValueError
        else:
            assert (gt_df["num_motors"] <= 1).all()

        logger.info("%s GROUND TRUTH DATAFRAME:\n%s", stage, gt_df)
        assert len(gt_df) == gt_df["tomo_id"].n_unique()
        self.gt_df = gt_df

        if stage == "train":
            pos_tomo_ids = []
            neg_tomo_ids = []
            pos_samples = []
            neg_samples = []
            for row_idx, row in enumerate(gt_df.iter_rows(named=True)):
                tomo_id = row["tomo_id"]
                ori_spacing = [row["voxel_spacing"]] * 3
                motor_zyxs = row["motor_zyx"]
                Z = row["Z"]
                ori_shape = (row["Z"], row["Y"], row["X"])
                radius_voxels = [1000.0 / e for e in ori_spacing]  # ZYX
                # sigma_voxels = [self.cfg.data.sigma * e for e in radius_voxels]  # ZYX

                if len(motor_zyxs) > 0:
                    pos_tomo_ids.append(tomo_id)
                    pos_zs = list(set([round(e[0]) for e in motor_zyxs]))
                    pos_samples.extend(
                        [
                            {
                                "type": "pos",
                                "tomo_id": tomo_id,
                                "ori_spacing": ori_spacing,
                                "ori_shape": ori_shape,
                                "ori_kpts": motor_zyxs,
                                "pos_z": z,
                            }
                            for z in pos_zs
                        ]
                    )
                else:
                    neg_tomo_ids.append(tomo_id)
                    pos_zs = []
                neg_zs = set(range(Z))
                for pos_z in pos_zs:
                    neg_zs = neg_zs.difference(
                        set(
                            range(
                                round(pos_z - radius_voxels[0]),
                                round(pos_z + radius_voxels[0]),
                            )
                        )
                    )
                neg_zs = list(neg_zs)
                neg_samples.append(
                    {
                        "type": "neg",
                        "tomo_id": tomo_id,
                        "ori_spacing": ori_spacing,
                        "ori_shape": ori_shape,
                        "ori_kpts": motor_zyxs,
                        "neg_zs": neg_zs,
                    }
                )
            self.pos_samples = pos_samples
            self.neg_samples = neg_samples
            self.num_neg_samples_per_epoch = round(
                len(self.pos_samples) * self.cfg.data.sampling.bg_ratio
            )
            logger.info(
                "Loaded %s dataset with %d pos samples (%d tomo), %d neg samples (%d tomo) per epoch",
                stage,
                len(self.pos_samples),
                len(pos_tomo_ids),
                self.num_neg_samples_per_epoch,
                len(neg_tomo_ids),
            )
        elif stage in ["val", "test"]:
            # augment to increase number of val/test samples
            # as a more robust way to track local CV metrics
            # with N enabled TTAs, number of validation samples
            # will be multiplied by N as well
            # TTA Transforms
            tta_dict_cfg = OmegaConf.to_object(cfg.data.tta)
            enable_ttas = tta_dict_cfg.pop("enable")
            if not isinstance(enable_ttas, list):
                assert isinstance(enable_ttas, str)
                enable_ttas = [enable_ttas]
            assert len(enable_ttas) == len(set(enable_ttas))

            tta_transforms = []
            tta_gt_dfs = []
            for _tta_idx, tta_name in enumerate(enable_ttas):
                tta_gt_df = gt_df.clone()
                tta_gt_df = tta_gt_df.with_columns(
                    pl.col("tomo_id").alias("ori_tomo_id"),
                    pl.col("tomo_id").map_elements(
                        lambda x: f"{x}@{tta_name}", return_dtype=pl.String
                    ),
                )
                tta_gt_dfs.append(tta_gt_df)
                tta_tf = build_tta(
                    tta_name,
                    weight=tta_dict_cfg.get(tta_name, 1.0),
                    ori_dims="yx",
                )
                tta_transforms.append((tta_name, tta_tf))
            tta_gt_df: pl.DataFrame = pl.concat(tta_gt_dfs)
            del tta_gt_dfs
            gc.collect()

            self.tta_gt_df = tta_gt_df.to_pandas()
            self.tta_transforms = tta_transforms
            self.num_ttas = len(tta_transforms)

            logger.info(
                "stage=%s, enable %d TTAs:\n%s",
                stage,
                len(tta_transforms),
                tta_transforms,
            )
            logger.info(
                "stage=%s load TTA augmented GT Dataframe with shape %s (ori %s)",
                stage,
                tta_gt_df.shape,
                gt_df.shape,
            )

            self.patch_item_metas = []
            self.all_tomos_meta = {}
            for row_idx, row in enumerate(gt_df.iter_rows(named=True)):
                tomo_id = row["tomo_id"]
                ori_spacing = [row["voxel_spacing"]] * 3
                ori_shape = [row["Z"], row["Y"], row["X"]]

                target_spacing_tomo_shape = tuple(
                    int(e * os / ts)
                    for e, os, ts in zip(ori_shape, ori_spacing, self.target_spacing)
                )
                target_spacing = self.target_spacing
                if cfg.data.agg_mode == "fit_single":
                    # use single patch for each slice using longest scaling + padding
                    r1 = max(
                        ori_spacing[1] / self.target_spacing[1],
                        ori_spacing[2] / self.target_spacing[2],
                    )
                    r2 = min(
                        cfg.data.patch_size[1] / ori_shape[1],
                        cfg.data.patch_size[2] / ori_shape[2],
                    )
                    if r1 > r2:
                        # accept increase voxel spacing (reduce resolution) on YX to improve runtime efficiency
                        # keep Z target spacing unchange
                        target_spacing_tomo_shape = (
                            int(ori_spacing[0] / self.target_spacing[0] * ori_shape[0]),
                            int(r2 * ori_shape[1]),
                            int(r2 * ori_shape[2]),
                        )
                        target_spacing = [
                            self.target_spacing[0],
                            ori_spacing[1] / r2,
                            ori_spacing[2] / r2,
                        ]

                # recheck to ensure everything is ok
                assert all(
                    [int(_ori_shape * _ori_spacing / _target_spacing) == _target_shape]
                    for _ori_shape, _ori_spacing, _target_spacing, _target_shape in zip(
                        ori_shape,
                        ori_spacing,
                        target_spacing,
                        target_spacing_tomo_shape,
                    )
                )

                # original keypoints in original tomogram voxel coordinate system (with original voxel spacing)
                ori_kpts = self.create_kpts_tensor(row["motor_zyx"], ori_spacing)
                patch_positions = get_sliding_patch_positions(
                    img_size=target_spacing_tomo_shape,
                    patch_size=cfg.data.patch_size,
                    border=cfg.data.border,
                    overlap=cfg.data.overlap,
                    start=cfg.data.start,
                    validate=False,
                )
                # print first sample for debuging purpose
                if row_idx == 0:
                    print(target_spacing_tomo_shape)
                    for _ps in patch_positions:
                        print(_ps[0], _ps[2])
                    print("------------------")
                patch_positions = torch.from_numpy(patch_positions)
                self.all_tomos_meta[tomo_id] = {
                    "ori_spacing": ori_spacing,
                    "target_spacing": target_spacing,
                    "ori_shape": ori_shape,
                    "ori_kpts": ori_kpts,
                    "tomo_shape": target_spacing_tomo_shape,
                    "patch_positions": patch_positions,
                }
                for tta_idx in range(self.num_ttas):
                    for i, patch_pos in enumerate(patch_positions):
                        self.patch_item_metas.append(
                            {
                                "tomo_idx": row_idx,
                                "tomo_id": tomo_id,
                                "ori_spacing": ori_spacing,
                                "target_spacing": target_spacing,
                                "ori_shape": ori_shape,
                                "tomo_shape": target_spacing_tomo_shape,
                                "patch_position": patch_pos,
                                "num_patches": len(patch_positions),
                                "ori_kpts": ori_kpts,
                                "tta_idx": tta_idx,
                                "is_first": i == 0,
                                "is_last": i == len(patch_positions) - 1,
                            }
                        )

            logger.info(
                "Loaded %s dataset with %d patches", stage, len(self.patch_item_metas)
            )

            # Start fast prefetch eval dataloader process
            self.shm_prefix = str(time.time()).replace(".", "")
            self.fast_val_tomo_loader = FastValTomoLoader(
                self.all_tomos_meta,
                data_dir=self.data_dir,
                io_backend="cv2",
                target_spacing=None,
                transform=None,  # using F.interpolate
                prefetch=cfg.data.fast_val_prefetch,
                # keep = (loader.prefetch_factor * loader.val_batch_size + 1) / (num_ttas * min_num_patches_per_tomo)
                # but this simple heuristic work fine
                keep=cfg.data.fast_val_prefetch + 3,
                interpolation_mode=cfg.data.transform.resample_mode,
                shm_prefix=self.shm_prefix,
                num_workers=cfg.data.fast_val_workers,
            )
        else:
            raise ValueError

    @property
    def worker_init_fn(self):
        def _woker_init_fn(worker_id: int, rank=None):
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset
            logger.debug(
                "Worker %d/%d: set MONAI transform random state to %d",
                worker_info.id + 1,
                worker_info.num_workers,
                worker_info.seed,
            )
            assert id(dataset) == id(self)

            # for val, set pytorch's number of threads to 1
            # otherwise, it's stuck..
            # ref: https://github.com/pytorch/pytorch/issues/89693
            if self.stage != "train":
                # print('{self.stage} OLD NUM THREADS:', torch.get_num_threads())
                torch.set_num_threads(1)

        return _woker_init_fn

    def create_kpts_tensor(self, motor_zyxs, ori_spacing):
        if len(motor_zyxs) > 0:
            kpts = torch.tensor(
                [
                    [
                        x,
                        y,
                        z,
                        (1000 * self.sigma / ori_spacing[2]) ** 2,  # cov_xx
                        (1000 * self.sigma / ori_spacing[1]) ** 2,  # cov_yy
                        (1000 * self.sigma / ori_spacing[0]) ** 2,  # cov_zz
                        0,  # cov_xy
                        0,  # cov_xz
                        0,  # cov_yz
                        0,  # class id
                    ]
                    for z, y, x in motor_zyxs
                ],
                dtype=torch.float32,
            )
        else:
            kpts = torch.zeros((0, 10))
        return kpts

    def __len__(self):
        if self.stage == "train":
            return (
                len(self.pos_samples) + self.num_neg_samples_per_epoch
            ) * self.cfg.data.dup_per_epoch
        else:
            return len(self.patch_item_metas)

    def _load_imgs(
        self, tomo_id: str, select_z: float, Z: int, ori_spacing: float
    ) -> List[np.ndarray]:
        """Load selected slice and the corresponding nearby slices.

        Args:
            tomo_id: tomogram id
            select_z: the selected Z coordinate of the central slice
            Z: tomogram's Z shape (total number of slices)
            ori_spacing: original spacing of the tomogram

        Returns:
            A list of slices/images
        """
        nearby_num = self.cfg.data.patch_size[0] - 1
        nearby_stride = self.cfg.data.transform.target_spacing[0] / ori_spacing[0]  # Z
        if nearby_num > 0:
            assert nearby_num % 2 == 0
            half = nearby_num // 2
            select_zs = [select_z + nearby_stride * i for i in range(-half, half + 1)]
            select_zs = [round(min(max(0, e), Z - 1)) for e in select_zs]
            assert len(select_zs) == nearby_num + 1
        else:
            select_zs = [round(select_z)]

        imgs = []
        if tomo_id.startswith("tomo_"):
            tomo_dir = os.path.join(self.data_dir, "raw", "train", tomo_id)
        else:
            tomo_dir = os.path.join(self.data_dir, "external", "tomogram", tomo_id)
        for z in select_zs:
            img = cv2.imread(
                os.path.join(tomo_dir, f"slice_{z:04d}.jpg"),
                cv2.IMREAD_UNCHANGED,
            )
            assert img.dtype == np.uint8
            if tomo_id in WRONG_QUANTILE_TOMO_IDS:
                img = img + 127
            # standardize to same target spacing
            img = cv2.resize(
                img,
                None,
                fx=ori_spacing[2] / self.target_spacing[2],
                fy=ori_spacing[1] / self.target_spacing[1],
                interpolation=CV2_INTERPOLATION_METHODS[
                    self.cfg.data.transform.resample_mode
                ],
            )
            imgs.append(img)
        return imgs

    def _get_train_item(self, idx, mix=False):
        assert not mix, "Mixup/Cutmix is currently not supported for 2D approach"
        idx = idx // self.cfg.data.dup_per_epoch
        if idx < len(self.pos_samples):
            # sample a positive
            sample = self.pos_samples[idx]
        else:
            # sample a negative
            sample = random.choice(self.neg_samples)
        tomo_id = sample["tomo_id"]
        ori_spacing = sample["ori_spacing"]
        Z, Y, X = sample["ori_shape"]
        radius_voxels = [1000.0 / e for e in ori_spacing]  # ZYX
        sigma_voxels = [self.cfg.data.sigma * e for e in radius_voxels]  # ZYX
        if sample["type"] == "pos":
            # if this sample is a positive one
            ori_select_z = sample["pos_z"]
            # sample center Z slice within truncated normal distribution
            select_z = round(
                truncnorm.rvs(
                    -self.cfg.data.sampling.rand_z_sigma_scale,
                    self.cfg.data.sampling.rand_z_sigma_scale,
                    loc=ori_select_z,
                    scale=sigma_voxels[0],
                )
            )
        elif sample["type"] == "neg":
            # elif this sample is a negative one
            if len(sample["neg_zs"]):
                select_z = ori_select_z = random.choice(sample["neg_zs"])
            else:
                select_z = ori_select_z = Z // 2
        else:
            raise ValueError
        ori_keypoints = np.array(sample["ori_kpts"]).reshape(-1, 3)  # ZYX

        # filter outside keypoints
        keypoints = ori_keypoints[
            np.abs(ori_keypoints[:, 0] - select_z)
            <= round(self.cfg.data.sampling.rand_z_sigma_scale * sigma_voxels[0]) + 1
        ]
        # FOR DEBUG PURPOSE ONLY
        # if sample["type"] == "pos":
        #     if keypoints.shape[0] == 0:
        #         print(
        #             f"POS ERROR:\n{ori_select_z} {select_z}\n{(self.cfg.data.sampling.rand_z_sigma_scale * sigma_voxels)}\n{ori_keypoints}"
        #         )
        # elif sample["type"] == "neg":
        #     if keypoints.shape[0] > 0:
        #         print(
        #             f"NEG ERROR:\n{ori_select_z} {select_z}\n{(self.cfg.data.sampling.rand_z_sigma_scale * sigma_voxels)}\n{keypoints}"
        #         )
        # else:
        #     raise ValueError

        # load image
        imgs = self._load_imgs(tomo_id, select_z, Z, ori_spacing)

        # annotated keypoints: x, y, scale, cov_xx, cov_yy, cov_xy, conf, class
        albu_kpts = np.full((keypoints.shape[0], 8), -1, dtype=np.float32)
        ori_cov_zyx = np.array(
            [
                [sigma_voxels[0] ** 2, 0, 0],
                [0, sigma_voxels[1] ** 2, 0],
                [0, 0, sigma_voxels[2] ** 2],
            ],
            dtype=np.float32,
        )  # ZYX
        spacing_scale = [
            tgt / ori for tgt, ori in zip(self.target_spacing, ori_spacing)
        ]  # ZYX
        for kpt_idx, zyx in enumerate(keypoints):
            conf_scale, mu_yx, cov_yx = z_slice_normal_dist_3d(
                mu_zyx=zyx, cov_zyx=ori_cov_zyx, z_slice=select_z
            )
            assert cov_yx[0, 1] == cov_yx[1, 0]  # 2x2 diagonal
            # albumentation keypoint: (x,y,scale)
            # Detectron2 adds 0.5 to COCO keypoint coordinates to convert them
            # from discrete pixel indices to floating point coordinates
            # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
            albu_kpts[kpt_idx, 0] = (mu_yx[1] + 0.5) / spacing_scale[2]  # X
            albu_kpts[kpt_idx, 1] = (mu_yx[0] + 0.5) / spacing_scale[1]  # Y
            albu_kpts[kpt_idx, 2] = 1.0  # scale used in albumentations
            albu_kpts[kpt_idx, 3] = cov_yx[1, 1] / (spacing_scale[2] ** 2)  # cov_xx
            albu_kpts[kpt_idx, 4] = cov_yx[0, 0] / (spacing_scale[1] ** 2)  # cov_yy
            albu_kpts[kpt_idx, 5] = cov_yx[0, 1] / (
                spacing_scale[2] * spacing_scale[1]
            )  # cov_yx
            albu_kpts[kpt_idx, 6] = conf_scale  # conf, max=1.0
            albu_kpts[kpt_idx, 7] = 0  # class

        # PASS THROUGHT ALBUMENTATION TRANSFORM
        data = {"keypoints": albu_kpts}
        img, data = self.do_albumentations_transform(imgs, data)
        assert (
            img.ndim == 3 and img.shape == self.cfg.data.patch_size
        ), f"{img.shape} {img.dtype} {self.cfg.data.patch_size} {sample['ori_shape']} {ori_spacing} {imgs[0].shape}"

        kpts = torch.from_numpy(data["keypoints"])
        assert kpts.dtype == torch.float32
        assert not np.isnan(kpts).any() and kpts.shape[1] == 8
        kpt_scales = kpts[:, 2:3]

        # convert from (x, y, scale, cov_xx, cov_yy, cov_xy, conf, class)
        # to (y, x, cov_yy, cov_xx, cov_xy, conf, class)
        kpts = kpts[:, [1, 0, 4, 3, 5, 6, 7]]
        # recompute the covariance matrix, given new scale
        kpts[:, 2:5] = kpts[:, 2:5] * (kpt_scales**2)

        # filter outside keypoints
        input_h, input_w = img.shape[1:]
        valid = torch.all(
            (kpts[:, :2] >= 0) & (kpts[:, :2] <= torch.tensor([[input_h, input_w]])),
            dim=1,
        )
        kpts = kpts[valid]

        if self.cfg.data.heatmap_same_sigma:
            # use the same sigma regardless of any augmentations
            sigma = torch.tensor(
                [
                    [
                        self.sigma * 1000.0 / self.target_spacing[1],
                        self.sigma * 1000.0 / self.target_spacing[2],
                    ]
                ]
            )  # yx, 1x2
            # (y, x, cov_yy, cov_xx, cov_xy, conf, class)
            kpts[:, 2:4] = sigma**2
            kpts[:, 4] = 0.0
        if self.cfg.data.heatmap_conf_scale_mode == "point":
            # use single point to represent a motor
            heatmap_gt = generate_2d_point_mask(
                heatmap_size=(1, *self.cfg.data.patch_size[1:]),
                keypoints=kpts,
                stride=self.cfg.data.heatmap_stride,
                dtype=torch.float16,
                lower=0.0,
                upper=1.0,
                add_offset=False,
            )
        else:
            # for gaussian heatmap or segmentation mask
            heatmap_gt = generate_2d_gaussian_heatmap(
                heatmap_size=(1, *self.cfg.data.patch_size[1:]),
                keypoints=kpts,
                stride=self.cfg.data.heatmap_stride,
                dtype=torch.float16,
                sigma_scale_factor=None,
                conf_interval=0.999,
                lower=0.0,
                upper=1.0,
                same_std=False,
                add_offset=False,
                conf_scale_mode=self.cfg.data.heatmap_conf_scale_mode,
                validate_cov_mat=True,
            )

        # 7 means (y, x, cov_yy, cov_xx, cov_xy, conf, class)
        assert kpts.shape[1] == 7
        if len(kpts) == 1:
            kpt_gt = torch.tensor(
                [[kpts[0, 0] / input_h, kpts[0, 1] / input_w]], dtype=torch.float32
            )  # (1, 2), normalized to range [0, 1]
            kptness_gt = kpts[0:1, 5]  # (1,)
            kpt_mask = torch.tensor([1], dtype=torch.float32)
        else:
            # just compute loss for kptness and heatmap
            # don't calculate loss for other components: kpt, dsnt
            kpt_gt = torch.tensor([[9876, 9876]], dtype=torch.float32)  # (1, 2)
            if len(kpts) == 0:
                kptness_gt = torch.tensor([0], dtype=torch.float32)  # (1,)
            elif len(kpts) > 1:
                kptness_gt = (
                    kpts[:, 5]
                    .max()
                    .reshape(
                        1,
                    )
                )  # (1,)
            else:
                raise AssertionError
            kpt_mask = torch.tensor([0], dtype=torch.float32)  # (1,)

        ret = {
            "idx": torch.tensor(idx, dtype=torch.long),
            "image": torch.from_numpy(img),  # CHW
            "heatmap_gt": heatmap_gt,  # CHW
            "kpt_gt": kpt_gt,
            "kptness_gt": kptness_gt,
            "kpt_mask": kpt_mask,
        }

        # debug
        if 0:
            from byu.utils.viz import create_heatmap_image, viz_fuse_image_and_heatmap

            viz_img = ret["image"].permute(1, 2, 0).numpy()
            viz_heatmap_01 = torch.nn.functional.interpolate(
                ret["heatmap_gt"][None], viz_img.shape[:2]
            )[0, 0].numpy()
            viz_heatmap = create_heatmap_image(viz_heatmap_01)
            viz_heatmap = cv2.cvtColor(viz_heatmap, cv2.COLOR_RGB2BGR)
            viz_fuse = viz_fuse_image_and_heatmap(
                viz_img, viz_heatmap_01, cmap_name="viridis"
            )
            # viz = np.concatenate([viz_fuse, viz_img, viz_heatmap], axis=1)
            viz = np.concatenate([viz_fuse, viz_heatmap], axis=0)
            text = f"max_conf={viz_heatmap_01.max():.2f}"
            cv2.putText(
                viz,
                text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            os.makedirs("outputs/tmp/", exist_ok=True)
            cv2.imwrite(f"outputs/tmp/{sample['type']}_{idx}.jpg", viz)
        return ret

    def _get_val_item(self, idx):
        patch_item_meta = self.patch_item_metas[idx]
        tomo_id = patch_item_meta["tomo_id"]
        ori_spacing = patch_item_meta["ori_spacing"]
        target_spacing = patch_item_meta["target_spacing"]
        ori_shape = patch_item_meta["ori_shape"]  # (Z, Y, X)
        tomo_shape = patch_item_meta["tomo_shape"]  # (Z, Y, X)
        patch_position = patch_item_meta["patch_position"]

        # the target-spaced tomo, not the original one
        tomo, _kpts, _, __ = self.fast_val_tomo_loader.load(tomo_id)
        assert tuple(tomo.shape) == tomo_shape and tomo.dtype == torch.uint8
        logger.debug(
            "Receive item idx=%d tomo_id=%s tomo=%s kpts=%s",
            idx,
            tomo_id,
            tomo.shape,
            _kpts.shape,
        )

        _roi_start, _roi_end, patch_start, patch_end = patch_position
        top_pad_z, top_pad_y, top_pad_x = [max(-start, 0) for start in patch_start]
        bot_pad_z, bot_pad_y, bot_pad_x = [
            max(0, end - size) for end, size in zip(patch_end, tomo_shape)
        ]
        assert top_pad_z == bot_pad_z == 0
        actual_crop_start = [max(0, start) for start in patch_start]
        actual_crop_end = [min(end, size) for end, size in zip(patch_end, tomo_shape)]
        crop_slices = tuple(
            slice(start, end) for start, end in zip(actual_crop_start, actual_crop_end)
        )
        crop = tomo[crop_slices]

        # pad if needed
        pad = (top_pad_x, bot_pad_x, top_pad_y, bot_pad_y, top_pad_z, bot_pad_z)
        if any(pad):
            crop = F.pad(crop, pad, mode="constant", value=0)
        assert crop.shape == self.cfg.data.patch_size
        # apply TTA
        # @TODO - TTA keypoints as well
        # WARNING: keypoints is not TTA-transformed, result in wrong keypoints relative to transformed image
        # We disable raw keypoints features temporarily
        tta_idx = patch_item_meta["tta_idx"]
        _tta_name, tta_tf = self.tta_transforms[tta_idx]
        crop, _weight = tta_tf.transform(crop)
        assert crop.dtype == torch.uint8

        return {
            "idx": torch.tensor(idx, dtype=torch.int64),
            "image": crop,
            # "num_kpts": torch.tensor(kpts.shape[0], dtype=torch.int64),
            # "kpts": pad_kpts(kpts, self.MAX_NUM_KPTS),
            "tomo_idx": torch.tensor(patch_item_meta["tomo_idx"], dtype=torch.int64),
            "ori_spacing": torch.tensor(ori_spacing, dtype=torch.float32),
            "target_spacing": torch.tensor(target_spacing, dtype=torch.float32),
            "patch_position": patch_position,
            "ori_shape": torch.tensor(ori_shape, dtype=torch.int16),
            "tomo_shape": torch.tensor(tomo_shape, dtype=torch.int16),  # spaced
            "tta_idx": torch.tensor(tta_idx, dtype=torch.int16),
            "patch_is_first": torch.tensor(
                patch_item_meta["is_first"], dtype=torch.bool
            ),
            "patch_is_last": torch.tensor(patch_item_meta["is_last"], dtype=torch.bool),
        }

    def __getitem__(self, idx):
        if self.stage == "train":
            return self._get_train_item(idx, mix=False)
        else:
            return self._get_val_item(idx)

    def build_train_transform(self):
        keypoint_params = A.KeypointParams(
            format="xys",
            remove_invisible=False,
            check_each_transform=False,
        )
        additional_targets = {
            f"image{i}": "image" for i in range(self.cfg.data.patch_size[0] - 1)
        }
        transform = A.Compose(
            [
                # flip
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # noise
                A.OneOf(
                    [
                        A.GaussNoise(
                            var_limit=(60, 120),
                            mean=0,
                            per_channel=True,
                            noise_scale_factor=0.5,
                            p=0.6,
                        ),
                        A.MultiplicativeNoise(
                            multiplier=(0.6, 1.4),
                            per_channel=True,
                            elementwise=True,
                            p=0.4,
                        ),
                    ],
                    p=0.1,
                ),
                # reduce quality
                A.OneOf(
                    [
                        # jitter on float32, implement AdaptiveDownscale based on current resolution
                        A.OneOf(
                            [
                                A.Downscale(
                                    scale_range=(0.5, 0.9),
                                    interpolation_pair={
                                        "upscale": cv2.INTER_LANCZOS4,
                                        "downscale": cv2.INTER_AREA,
                                    },
                                    p=0.1,
                                ),
                                A.Downscale(
                                    scale_range=(0.5, 0.9),
                                    interpolation_pair={
                                        "upscale": cv2.INTER_LINEAR,
                                        "downscale": cv2.INTER_AREA,
                                    },
                                    p=0.1,
                                ),
                                A.Downscale(
                                    scale_range=(0.5, 0.9),
                                    interpolation_pair={
                                        "upscale": cv2.INTER_LINEAR,
                                        "downscale": cv2.INTER_LINEAR,
                                    },
                                    p=0.8,
                                ),
                            ],
                            p=0.6,
                        ),
                        A.ImageCompression(
                            compression_type="jpeg", quality_range=(20, 80), p=0.3
                        ),
                        A.Posterize(num_bits=(4, 6), p=0.1),
                    ],
                    p=0.25,
                ),
                # texture, contrast
                A.OneOf(
                    [
                        # wrong on float32 img
                        A.Emboss(alpha=(0.3, 0.6), strength=(0.2, 0.8), p=0.4),
                        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.0, 0.4), p=0.5),
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(16, 16), p=0.1),
                    ],
                    p=0.1,
                ),
                # color
                A.OneOf(
                    [
                        A.OneOf(
                            [
                                A.RandomBrightnessContrast(
                                    brightness_limit=(-0.3, 0.0),
                                    contrast_limit=(-0.2, 0.0),
                                    brightness_by_max=False,
                                    p=0.4,
                                ),
                                A.RandomBrightnessContrast(
                                    brightness_limit=(0.0, 0.4),
                                    contrast_limit=(-0.5, 0.0),
                                    brightness_by_max=False,
                                    p=0.4,
                                ),
                                A.RandomBrightnessContrast(
                                    brightness_limit=(-0.3, 0.0),
                                    contrast_limit=(0.0, 0.5),
                                    brightness_by_max=False,
                                    p=0.1,
                                ),
                                A.RandomBrightnessContrast(
                                    brightness_limit=(0.0, 0.3),
                                    contrast_limit=(0.0, 0.5),
                                    brightness_by_max=False,
                                    p=0.1,
                                ),
                            ],
                            p=0.4,
                        ),
                        A.RandomToneCurve(scale=0.3, per_channel=True, p=0.4),
                        A.RandomGamma(gamma_limit=(60, 150), p=0.2),
                    ],
                    p=0.5,
                ),
                # geometric
                A.OneOf(
                    [
                        # strong rotate
                        A.Affine(
                            scale={"x": (0.6, 1.2), "y": (0.6, 1.2)},
                            translate_percent=None,
                            rotate=(0, 360),
                            shear={"x": (-5, 5), "y": (-5, 5)},
                            interpolation=cv2.INTER_LINEAR,
                            cval=128,
                            mode=cv2.BORDER_CONSTANT,
                            fit_output=True,
                            keep_ratio=False,
                            balanced_scale=False,
                            p=0.4,
                        ),
                        # strong shear
                        A.Affine(
                            scale={"x": (0.6, 1.2), "y": (0.6, 1.2)},
                            translate_percent=None,
                            rotate=(0, 360),
                            shear={"x": (-20, 20), "y": (-20, 20)},
                            interpolation=cv2.INTER_LINEAR,
                            cval=128,
                            mode=cv2.BORDER_CONSTANT,
                            fit_output=True,
                            keep_ratio=False,
                            balanced_scale=False,
                            p=0.3,
                        ),
                        A.Perspective(
                            scale=(0.05, 0.12),
                            keep_size=False,
                            pad_mode=cv2.BORDER_CONSTANT,
                            pad_val=128,
                            fit_output=True,
                            interpolation=cv2.INTER_LINEAR,
                            p=0.3,
                        ),
                        # very slow + in-accurate (approximate) keypoints
                        # how to deal with out-of-image keypoints?
                        # this break intrinsic contrains -> not well-combined with other Dropout augs
                        # A.OneOf([
                        #     A.PiecewiseAffine(scale=(0.025, 0.025), nb_rows=4, nb_cols=4, interpolation=cv2.INTER_LINEAR, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, keypoints_threshold=0.01, p=0.33),
                        #     A.PiecewiseAffine(scale=(0.016, 0.016), nb_rows=6, nb_cols=6, interpolation=cv2.INTER_LINEAR, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, keypoints_threshold=0.01, p=0.33),
                        #     A.PiecewiseAffine(scale=(0.0125, 0.0125), nb_rows=8, nb_cols=8, interpolation=cv2.INTER_LINEAR, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, keypoints_threshold=0.01, p=0.34)
                        # ], p=0.1)
                    ],
                    p=0.8,
                ),
                # crop go after geometric to prevent too much information loss
                AC.CustomRandomSizedBBoxSafeCrop(
                    crop_size=self.cfg.data.patch_size[1:],
                    scale=(0.25, 1.0),  # unused
                    ratio=(0.25, 1.5),  # unused
                    get_bbox_func=partial(
                        _byu_get_safe_bbox,
                        margin_xy=(
                            1000.0 / self.target_spacing[2],
                            1000.0 / self.target_spacing[1],
                        ),
                    ),
                    retry=10,
                    p=1.0,
                ),
                A.PadIfNeeded(
                    self.cfg.data.patch_size[1],
                    self.cfg.data.patch_size[2],
                    position="top_left",
                    border_mode=cv2.BORDER_CONSTANT,
                    value=128,
                    p=1.0,
                ),
                # dropout
                A.OneOf(
                    [
                        AC.CustomCoarseDropout(
                            fill_value=0,
                            num_holes_range=(4, 8),
                            hole_height_range=(0.1, 0.2),
                            hole_width_range=(0.1, 0.2),
                            p=0.5,
                        ),
                        AC.CustomGridDropout(
                            ratio=0.5,
                            random_offset=True,
                            fill_value=0,
                            holes_number_xy=((3, 8), (3, 8)),
                            p=0.3,
                        ),
                        AC.CustomXYMasking(
                            num_masks_x=(2, 5),
                            num_masks_y=(2, 5),
                            mask_x_length=(0.03, 0.05),
                            mask_y_length=(0.03, 0.05),
                            fill_value=0,
                            p=0.2,
                        ),
                    ],
                    p=self.cfg.data.augment.p_dropout,
                ),
            ],
            keypoint_params=keypoint_params,
            additional_targets=additional_targets,
            p=1.0,
        )
        logger.info("%s transform:\n%s", self.stage, transform)
        return transform

    def do_albumentations_transform(self, imgs, data):
        """
        Apply Albumentations transform to a list of images with same random parameters
        compute from the center slice (key frame).
        """
        num_imgs = len(imgs)
        assert num_imgs == self.cfg.data.patch_size[0]
        center_idx = num_imgs // 2
        img_param_keys = [f"image{i}" for i in range(num_imgs - 1)]
        img_param_keys.insert(center_idx, "image")
        data.update({k: v for k, v in zip(img_param_keys, imgs)})
        data = self.transform(**data)
        imgs = [data.pop(k) for k in img_param_keys]
        if num_imgs == 1:
            img = imgs[0][None]  # 1HW
        else:
            img = np.stack(imgs, axis=0)  # CHW
        return img, data


# FOR TESTING PURPOSE
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from omegaconf import OmegaConf

    yaml_str = """

cv:
    strategy: skf4_rd42
    num_folds: 4
    fold_idx: 0

env:
    data_dir: '/home/dangnh36/datasets/.comp/byu/'

data:
    _target_: yagm.data.base_datamodule.BaseDataModule
    _dataset_target_: byu.data.datasets.dataset_2d.BYU2dDataset

    patch_size: [3, 768, 768]
    border: [0, 0, 0]
    overlap: [0, 0, 0]
    start: [0, 0, 0]
    fast_val_workers: 1
    fast_val_prefetch: 2
    dup_per_epoch: 100

    # agg_mode: patch
    agg_mode: fit_single

    heatmap_stride: [4,4]
    sigma: 0.2

    sampling:
        bg_ratio: 0.1
        rand_z_sigma_scale: 1.0

    transform:
        target_spacing: [64,16,16]
        resample_mode: trilinear
        equalize: False

    augment:
        p: 0.8
        p_dropout: 0.25

    tta:
        enable: [yx, xy, yx_x, yx_y]
"""
    from lightning import seed_everything

    STAGE = "val"
    seed_everything(42)
    global_cfg = OmegaConf.create(yaml_str)
    dataset = Heatmap2dDataset(cfg=global_cfg, stage=STAGE)
    if STAGE == "val":
        dataset.fast_val_tomo_loader.start()

    print("DATASET LEN:", len(dataset))

    for idx in range(0, len(dataset), 5):
        sample = dataset[idx]
        print(idx, {k: [v.shape, v.dtype] for k, v in sample.items()})
