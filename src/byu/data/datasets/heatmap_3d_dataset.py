import gc
import json
import logging
import math
import os
import pprint
import random
import time
from math import pi
from typing import List, Tuple

import numpy as np
import polars as pl
import torch
from monai import transforms as T
from omegaconf import OmegaConf
from torch.nn import functional as F
from tqdm import tqdm
from yagm.data.datasets.base_dataset import BaseDataset
from yagm.transforms import monai_custom as CT
from yagm.transforms.keypoints.helper import kpts_spatial_crop
from yagm.transforms.sliding_window import get_sliding_patch_positions
from yagm.transforms.tta_3d import build_tta

from byu.data.fast_val_loader import FastValTomoLoader
from byu.data.io import (
    MultithreadOpencvTomogramLoader,
    OpencvTomogramLoader,
    load_spaced_val_sample_cache,
    read_tomo,
)
from byu.utils.misc import compute_target_spacing_shape, generate_heatmap

logger = logging.getLogger(__name__)


ALL_TRAIN_TTAS = [
    "zyx",
    "zyx_x",
    "zyx_y",
    "zyx_z",
    "zyx_xy",
    "zyx_xz",
    "zyx_yz",
    "zyx_xyz",
    "zxy",
    "zxy_x",
    "zxy_y",
    "zxy_z",
    "zxy_xy",
    "zxy_xz",
    "zxy_yz",
    "zxy_xyz",
]

INTERPOLATE_TO_GRID_RESAMPLE_MODE_MAP = {
    "trilinear": "bilinear",
    "nearest": "nearest",
    "area": "area",
    "bicubic": "bicubic",
}


def build_val_transform(
    global_cfg, image_keys=["image"], label_keys=["heatmap"], log_stats=False
):
    lazy = global_cfg.data.transform.lazy
    align_corners = False
    resample_mode = INTERPOLATE_TO_GRID_RESAMPLE_MODE_MAP[
        global_cfg.data.transform.resample_mode
    ]
    keys = image_keys + label_keys

    device = global_cfg.data.transform.device
    enable_mp_cuda_resample = (
        device is not None
        and "cuda" in device
        and global_cfg.loader.val_num_workers > 0
    )
    if enable_mp_cuda_resample:
        logger.warning(
            "Forcing Torch Multiprocessing start method from %s to `spawn`",
            torch.multiprocessing.get_start_method(),
        )
        torch.multiprocessing.set_start_method("spawn", force=True)
    _overrides = {
        "mode": resample_mode,
        "padding_mode": "zeros",
        "align_corners": align_corners,
        "resample_mode": resample_mode,
        "device": device,
        "dtype": "float32",
    }
    overrides = {k: _overrides for k in keys}

    val_transforms = [
        CT.CustomSpacingd(
            keys=keys,
            spacing_key="spacing_scale",
            mode=resample_mode,
            padding_mode="zeros",
            align_corners=align_corners,
            dtype=None,
            scale_extent=False,
            lazy=lazy,
        ),
        # CT.CropBySlicesd(keys=keys, slices_key="crop_slices", lazy=lazy),
        CT.ApplyTransformToNormalDistributionsd(
            keys=["kpts"],
            refer_keys=["image"],
            dtype=torch.float32,
            affine=None,
            invert_affine=True,
        ),
    ]
    if enable_mp_cuda_resample:
        # tensors must be in same device across processes to be collated
        val_transforms.append(T.ToDeviced(keys=keys, device="cpu"))
    logger.info("VAL TRANSFORMS:\n%s", pprint.pformat(val_transforms, indent=4))
    return T.Compose(
        val_transforms,
        map_items=True,
        log_stats=log_stats,
        lazy=lazy,
        overrides=overrides,
    )


def build_train_transform(
    global_cfg, image_keys=["image"], label_keys=["heatmap"], log_stats=False
):
    keys = image_keys + label_keys
    lazy = global_cfg.data.transform.lazy
    acfg = global_cfg.data.aug
    target_spacing = global_cfg.data.transform.target_spacing
    resample_mode = INTERPOLATE_TO_GRID_RESAMPLE_MODE_MAP[
        global_cfg.data.transform.resample_mode
    ]
    align_corners = False
    device = global_cfg.data.transform.device

    # validate config
    if acfg.rand_shift:
        assert global_cfg.data.sampling.method == "rand_crop" and not (
            global_cfg.data.sampling.rand_crop.random_center
            or global_cfg.data.sampling.rand_crop.auto_correct_center
        )

    # resample on CUDA or CPU
    enable_mp_cuda_resample = (
        device is not None
        and "cuda" in device
        and global_cfg.loader.train_num_workers > 0
    )
    if enable_mp_cuda_resample:
        logger.warning(
            "Forcing Torch Multiprocessing start method from `%s` to `spawn`",
            torch.multiprocessing.get_start_method(),
        )
        torch.multiprocessing.set_start_method("spawn", force=True)

    # overrides: this optional parameter allows you to specify a dictionary of parameters that should be overridden
    # when executing a pipeline. These each parameter that is compatible with a given transform is then applied
    # to that transform before it is executed. Note that overrides are currently only applied when
    # :ref:`Lazy Resampling<lazy_resampling>` is enabled for the pipeline or a given transform. If lazy is False
    # they are ignored. Currently supported args are:
    # {``"mode"``, ``"padding_mode"``, ``"dtype"``, ``"align_corners"``, ``"resample_mode"``, ``device``}.
    _overrides = {
        "mode": resample_mode,
        "padding_mode": "zeros",
        "align_corners": align_corners,
        "resample_mode": resample_mode,
        "device": device,
        "dtype": "float32",
    }
    overrides = {k: _overrides for k in keys}
    D, W, H = global_cfg.data.patch_size

    def _Compose(transforms):
        return T.Compose(
            transforms,
            map_items=True,
            log_stats=log_stats,
            lazy=lazy,
            overrides=overrides,
        )

    def _OneOf(transforms, weights):
        # assert len(transforms) == len(weights) and abs(sum(weights) - 1.0) < 1e-6
        assert len(transforms) == len(weights)
        if sum(weights) == 0:
            logger.warning(
                "MONAI OneOf was init with weights of all 0.0, set to all 1.0 instead:\n%s",
                pprint.pformat(transforms, indent=4),
            )
            weights = [1.0] * len(weights)
        return T.OneOf(
            transforms,
            weights=weights,
            lazy=lazy,
            overrides=overrides,
        )

    # SPACING
    spacing_transforms = [
        CT.CustomSpacingd(
            keys=keys,
            spacing_key="spacing_scale",
            mode=resample_mode,
            padding_mode="zeros",
            align_corners=False,
            dtype=None,
            scale_extent=False,
            lazy=lazy,
        )
    ]

    # CROP
    if global_cfg.data.sampling.method == "pre_patch":
        # oversample foreground patches
        crop_transforms = [
            CT.CropBySlicesd(keys=keys, slices_key="crop_slices", lazy=lazy)
        ]
    elif global_cfg.data.sampling.method == "rand_crop":
        # use normal Random Crop
        # crop_transforms = [
        #     T.RandSpatialCropd(
        #         keys=keys,
        #         roi_size=(H, W, D),
        #         max_roi_size=None,
        #         random_center=True,
        #         random_size=False,
        #         lazy=lazy,
        #     )
        # ]
        crop_transforms = [
            CT.RandSpatialCropByKeypointsd(
                keys=keys,
                keypoints_key="spaced_kpts",
                roi_size=(H, W, D),
                max_roi_size=None,
                random_center=global_cfg.data.sampling.rand_crop.random_center,
                random_size=False,
                margin=global_cfg.data.sampling.rand_crop.margin,
                auto_correct_center=global_cfg.data.sampling.rand_crop.auto_correct_center,
                allow_missing_keys=False,
                lazy=lazy,
            )
        ]
    else:
        raise ValueError

    # SPATIAL TRANSFORM
    spatial_transforms = [
        # ROT90 along XY/HW plane: HWD -> WHD
        # T.RandRotate90d(keys=keys, prob=0.5, max_k=1, spatial_axes=(0, 1), lazy=lazy),
        # FLIP
        # 8 posible transforms: Identity, Flip_X, Flip_Y, Flip_Z, rot180_X, rot180_Y, rot180_Y, Flip_XYZ
        # T.RandFlipd(keys=keys, prob=0.5, spatial_axis=0, lazy=lazy),
        # T.RandFlipd(keys=keys, prob=0.5, spatial_axis=1, lazy=lazy),
        # T.RandFlipd(keys=keys, prob=0.5, spatial_axis=2, lazy=lazy),
        # # Rotate 180, no need since two flip along X,Y is equivalent to rotate 180 around Z
        # CT.RandRotate180d(keys=keys, prob=0.5, spatial_axes=(0, 1), lazy=lazy),
        # CT.RandRotate180d(keys=keys, prob=0.5, spatial_axes=(0, 2), lazy=lazy),
        # CT.RandRotate180d(keys=keys, prob=0.5, spatial_axes=(1, 2), lazy=lazy),
        # ZOOM, >1 is zoom in, <1 is zoom out
        T.RandZoomd(
            keys=keys,
            prob=acfg.zoom_prob,
            min_zoom=(
                acfg.zoom_range[0][0],
                acfg.zoom_range[1][0],
                acfg.zoom_range[2][0],
            ),
            max_zoom=(
                acfg.zoom_range[0][1],
                acfg.zoom_range[1][1],
                acfg.zoom_range[2][1],
            ),
            mode="bilinear",
            padding_mode="constant",
            align_corners=align_corners,
            dtype=None,
            keep_size=True,
            lazy=lazy,
        ),
        # Affine
        _OneOf(
            [
                # NOTE: Affine: rotate -> shear -> translate -> scale
                # Affine 1: slight, focus on XY only
                T.RandAffined(
                    keys=keys,
                    prob=acfg.affine1_prob + acfg.affine2_prob,
                    rotate_range=((0, 0), (0, 0), (0, 2 * pi)),
                    shear_range=None,
                    translate_range=None,
                    # > 0 is zoom out, <0 is zoom in
                    scale_range=(
                        (-acfg.affine1_scale, acfg.affine1_scale),
                        (-acfg.affine1_scale, acfg.affine1_scale),
                        (-acfg.affine1_scale, acfg.affine1_scale),
                    ),
                    spatial_size=None,
                    mode="bilinear",  # bilinear, nearest
                    padding_mode="constant",
                    cache_grid=True,
                    lazy=lazy,
                ),
                # Affine 2: heavy, focus on all dims
                T.RandAffined(
                    keys=keys,
                    prob=acfg.affine1_prob + acfg.affine2_prob,
                    rotate_range=(
                        (
                            -math.radians(acfg.affine2_rotate_xy),
                            math.radians(acfg.affine2_rotate_xy),
                        ),
                        (
                            -math.radians(acfg.affine2_rotate_xy),
                            math.radians(acfg.affine2_rotate_xy),
                        ),
                        (0, 2 * pi),
                    ),
                    shear_range=(
                        (-acfg.affine2_shear, acfg.affine2_shear),
                        (-acfg.affine2_shear, acfg.affine2_shear),
                        (-acfg.affine2_shear, acfg.affine2_shear),
                    ),
                    translate_range=None,
                    # > 0 is zoom out, <0 is zoom in
                    scale_range=(
                        (-acfg.affine2_scale, acfg.affine2_scale),
                        (-acfg.affine2_scale, acfg.affine2_scale),
                        (-acfg.affine2_scale, acfg.affine2_scale),
                    ),
                    spatial_size=None,
                    mode="bilinear",  # bilinear, nearest
                    padding_mode="constant",
                    cache_grid=True,
                    lazy=lazy,
                ),
            ],
            weights=[acfg.affine1_prob, acfg.affine2_prob],
        ),
    ]
    if acfg.rand_shift:
        # add random shift
        # very useful if random crop and random_center=auto_correct_center=False
        expect_avg_radius = [
            1000.0 / e for e in global_cfg.data.transform.target_spacing
        ]
        spatial_transforms.append(
            T.RandAffined(
                keys=keys,
                prob=1.0,
                translate_range=(
                    (
                        -max(0, H / 2 - expect_avg_radius[1]),
                        max(0, H / 2 - expect_avg_radius[1]),
                    ),
                    (
                        -max(0, W / 2 - expect_avg_radius[2]),
                        max(0, W / 2 - expect_avg_radius[2]),
                    ),
                    (
                        -max(0, D / 2 - expect_avg_radius[0] / 4),
                        max(0, D / 2 - expect_avg_radius[0] / 4),
                    ),
                ),
                spatial_size=None,
                mode="bilinear",  # bilinear, nearest
                padding_mode="constant",
                cache_grid=True,
                lazy=lazy,
            )
        )

    # NO LAZY SPATIAL TRANSFORM
    _no_lazy_spatial_transforms = [
        T.RandGridDistortiond(
            keys=keys,
            num_cells=(H // 32, W // 32, D // 32),
            prob=acfg.grid_distort_prob,
            distort_limit=(-0.15, 0.15),
            mode="bilinear",
            padding_mode="constant",
        ),
        # T.Rand3DElasticd(
        #     keys=keys,
        #     sigma_range=(11, 11),
        #     magnitude_range=(5, 5),
        #     prob=?,
        #     spatial_size=None,
        #     mode="bilinear",
        #     padding_mode="constant",
        # ),
        # NOTE: This transform could make particle's center float around
        # also could break heatmap volume's intrinsic properties
        T.RandSmoothDeformd(
            keys=keys,
            spatial_size=(H, W, D),
            rand_size=(H // 8, W // 8, D // 8),
            pad=0,
            field_mode="area",
            align_corners=None,
            prob=acfg.smooth_deform_prob,
            # 6 is min of particle radius (apo-ferritin)
            def_range=(-4 / max(H, W, D), 4 / max(H, W, D)),
            grid_mode="nearest",
            grid_padding_mode="zeros",
            grid_align_corners=align_corners,
        ),
    ]

    intensity_transforms = [
        _OneOf(
            [
                # mean shift + std scale
                # (1-x)(1-y) = 1 - p
                # for simple, we select x=y= 1 - (1-p)**0.5
                _Compose(
                    [
                        # mean shift
                        _OneOf(
                            [
                                T.RandShiftIntensityd(
                                    image_keys,
                                    offsets=(-40, 80),
                                    safe=False,
                                    prob=1.0 - (1.0 - acfg.intensity_prob) ** 0.5,
                                    channel_wise=True,
                                ),
                                T.RandStdShiftIntensityd(
                                    image_keys,
                                    (-0.7, 1.2),
                                    prob=1.0 - (1.0 - acfg.intensity_prob) ** 0.5,
                                    nonzero=False,
                                    channel_wise=True,
                                ),
                            ],
                            weights=[0.5, 0.5],
                        ),
                        # std/contrast scale (multiplicative)
                        _OneOf(
                            [
                                # decrease std or contrast -> harder sample -> higher prob
                                T.RandScaleIntensityFixedMeand(
                                    image_keys,
                                    prob=1.0 - (1.0 - acfg.intensity_prob) ** 0.5,
                                    factors=(-0.6, 0.0),
                                    fixed_mean=True,
                                    preserve_range=False,
                                ),
                                # increase std or contrast -> easier sample -> lower prob
                                T.RandScaleIntensityFixedMeand(
                                    image_keys,
                                    prob=1.0 - (1.0 - acfg.intensity_prob) ** 0.5,
                                    factors=(0.0, 0.8),
                                    fixed_mean=True,
                                    preserve_range=False,
                                ),
                            ],
                            weights=[0.7, 0.3],
                        ),
                    ]
                ),
                # mean/std multiplicative x*(1+factor)
                T.RandScaleIntensityd(
                    image_keys,
                    factors=(-0.7, 0.7),
                    prob=acfg.intensity_prob,
                    channel_wise=True,
                ),
                # mean/std polynomial x**gamma
                T.RandAdjustContrastd(
                    image_keys,
                    prob=acfg.intensity_prob,
                    gamma=(0.25, 1.75),
                    invert_image=False,
                    retain_stats=False,
                ),
                # histogram modification
                T.RandHistogramShiftd(
                    image_keys, num_control_points=(10, 20), prob=acfg.intensity_prob
                ),
            ],
            weights=[0.4, 0.2, 0.3, 0.1],
        ),
        T.RandGaussianSmoothd(
            image_keys,
            sigma_x=(0.5, 4.5),
            sigma_y=(0.5, 4.5),
            sigma_z=(0.5, 4.5),
            prob=acfg.smooth_prob,
            approx="erf",
        ),
        # CT.RandMedianSmoothd(image_keys, radius = 1, prob = 0.0) # slow on CPU, radius >=2 -> large RAM
    ]
    if acfg.hist_equalize:
        intensity_transforms.append(
            T.HistogramNormalized(image_keys, num_bins=256, min=0, max=255.0, mask=None)
        )

    # POINTS Transform
    keypoint_transform = CT.ApplyTransformToNormalDistributionsd(
        keys=["kpts"],
        refer_keys=image_keys[0],
        dtype=torch.float32,
        affine=None,
        invert_affine=True,
    )

    # mode: nearest, nearest-exact, linear, bilinear, bicubic, trilinear, area
    _downsample_config = [
        ("trilinear", "trilinear", 0.4),
        ("trilinear", "nearest", 0.1),
        ("area", "trilinear", 0.1),
        ("area", "nearest", 0.1),
        ("nearest", "trilinear", 0.2),
        ("nearest", "nearest", 0.1),
    ]
    downsample_transforms = [
        _OneOf(
            [
                T.RandSimulateLowResolutiond(
                    keys=keys,
                    prob=acfg.downsample_prob,
                    downsample_mode=e[0],
                    upsample_mode=e[1],
                    zoom_range=(0.6, 0.9),
                    align_corners=None,
                )
                for e in _downsample_config
            ],
            weights=[e[2] for e in _downsample_config],
        )
    ]

    # Dropout Transform
    dropout_transforms = [
        CT.RandCoarseDropoutWithKeypointsd(
            keys=image_keys,
            keypoints_key="kpts",
            holes=3,
            spatial_size=(
                500 // target_spacing[2],
                500 // target_spacing[1],
                500 // target_spacing[0],
            ),
            dropout_holes=True,
            fill_value=(0, 255),
            max_holes=8,
            max_spatial_size=(
                2000 // target_spacing[2],
                2000 // target_spacing[1],
                2000 // target_spacing[0],
            ),
            remove="patch",
            keypoint_margins="auto",
            max_retries=10,
            prob=acfg.coarse_dropout_prob,
        )
    ]

    if global_cfg.data.io_backend == "cache":
        train_transforms = [
            *spatial_transforms,
            # *no_lazy_spatial_transforms,
            *intensity_transforms,
            *downsample_transforms,
            keypoint_transform,
            *dropout_transforms,
        ]
    elif global_cfg.data.aug.enable:
        train_transforms = [
            *spacing_transforms,
            *crop_transforms,
            *spatial_transforms,
            # *no_lazy_spatial_transforms,
            *intensity_transforms,
            *downsample_transforms,
            keypoint_transform,
            *dropout_transforms,
        ]
    else:
        train_transforms = [
            *spacing_transforms,
            *crop_transforms,
            keypoint_transform,
        ]

    if enable_mp_cuda_resample:
        # tensors must be in same device across processes to be collated
        train_transforms.append(T.ToDeviced(keys=keys, device="cpu"))
    logger.info("TRAIN TRANSFORMS:\n%s", train_transforms)
    return _Compose(train_transforms)


class Heatmap3dDataset(BaseDataset):
    """
    Dataset for 3D Heatmap Estimation approach,
    typically used in combination with a 3D UNet model
    """

    MAX_NUM_KPTS = 32

    def __init__(self, cfg, stage="train", cache=None):
        del cache
        # by default, Pytorch use multiple threads to speedup CPU operation
        # if not set number of threads to 1 before `fork` caused by Dataloader with num_workers > 0
        # then boom, segmentfault, related to: https://github.com/pytorch/pytorch/issues/54752
        # each dataloader process then can change the number of threads by a call to torch.set_num_threads()
        # e.g, in worker_init_fn passed to DataLoader
        logger.info(
            "Temporarily set Pytorch number of threads from %d to 1",
            torch.get_num_threads(),
        )
        torch.set_num_threads(1)
        super().__init__(cfg, stage)
        data_dir = cfg.env.data_dir
        self.data_dir = data_dir

        self.target_spacing = cfg.data.transform.target_spacing
        self.sigma = cfg.data.sigma
        self.heatmap_stride = cfg.data.transform.heatmap_stride

        self.heatmap_mode = cfg.data.transform.heatmap_mode
        self.io_backend = cfg.data.io_backend
        self.tomo_reader = None

        # LOAD LABELS
        gt_df = (
            pl.scan_csv(
                os.path.join(
                    data_dir,
                    "processed",
                    f"{cfg.data.label_fname}.csv" if stage == "train" else "gt_v3.csv",
                )
            )
            .sort("tomo_id")
            .with_columns(
                pl.col("motor_zyx").map_elements(eval, return_dtype=pl.Object),
                pl.struct(["Z", "Y", "X", "voxel_spacing"])
                .map_elements(
                    lambda e: compute_target_spacing_shape(
                        (e["Z"], e["Y"], e["X"]),
                        [e["voxel_spacing"]] * 3,
                        self.target_spacing,
                        method="torch" if cfg.data.io_backend == "cache" else "monai",
                        scale_extent=False,
                    ),
                    return_dtype=pl.Object,
                )
                .alias("target_spacing_tomo_shape"),
            )
            .collect()
        )
        all_tomo_ids = sorted(gt_df["tomo_id"].unique().to_list())

        # CV Splitting
        with open(
            os.path.join(data_dir, "processed", "cv", "v3", f"{cfg.cv.strategy}.json"),
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

        # Generate list of patches
        self.patch_item_metas = []
        self.tomo_ids = []
        self.all_tomos_meta = {}
        if stage in ["val", "test"]:
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
                    ori_dims="zyx",
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

            for row_idx, row in enumerate(gt_df.iter_rows(named=True)):
                tomo_id = row["tomo_id"]
                self.tomo_ids.append(tomo_id)
                ori_spacing = [row["voxel_spacing"]] * 3
                ori_shape = [row["Z"], row["Y"], row["X"]]
                target_spacing_tomo_shape = row["target_spacing_tomo_shape"]
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
                patch_positions = torch.from_numpy(patch_positions)
                self.all_tomos_meta[tomo_id] = {
                    "ori_spacing": ori_spacing,
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
        elif stage == "train":
            # TTA
            self.tta_transforms = [
                (
                    tta_name,
                    build_tta(
                        tta_name,
                        weight=1.0,
                        ori_dims="zyx",
                    ),
                )
                for tta_name in ALL_TRAIN_TTAS
            ]
            self.num_ttas = len(self.tta_transforms)

            rand_crop_pos_tomo_ids = []
            rand_crop_neg_tomo_ids = []
            for row_idx, row in tqdm(
                enumerate(gt_df.iter_rows(named=True)),
                total=len(gt_df),
                desc="Pre-patching..",
            ):
                tomo_id = row["tomo_id"]
                self.tomo_ids.append(tomo_id)
                ori_spacing = [row["voxel_spacing"]] * 3
                ori_shape = [row["Z"], row["Y"], row["X"]]
                target_spacing_tomo_shape = row["target_spacing_tomo_shape"]
                ori_kpts = self.create_kpts_tensor(row["motor_zyx"], ori_spacing)
                tomo_meta = {
                    "ori_spacing": ori_spacing,
                    "ori_shape": [row["Z"], row["Y"], row["X"]],
                    "ori_kpts": ori_kpts,
                    "tomo_shape": target_spacing_tomo_shape,
                }
                self.all_tomos_meta[tomo_id] = tomo_meta
                if cfg.data.sampling.method == "pre_patch":
                    patch_positions = get_sliding_patch_positions(
                        img_size=target_spacing_tomo_shape,
                        patch_size=cfg.data.patch_size,
                        border=cfg.data.border,
                        overlap=cfg.data.sampling.pre_patch.overlap,
                        start=(0, 0, 0),
                        validate=False,
                    )
                    patch_positions = torch.from_numpy(patch_positions)
                    tomo_meta["patch_positions"] = patch_positions

                    for patch_pos in patch_positions:
                        patch_start, patch_end, crop_start, crop_end = patch_pos
                        # just (N, 3), represent ZYX in target spacing coordinate system
                        target_spacing_kpts = (
                            ori_kpts[:, [2, 1, 0]].clone()
                            * torch.tensor([ori_spacing])
                            / torch.tensor([self.target_spacing])
                        )  # ZYX, just approximation
                        _cropped_kpts = kpts_spatial_crop(
                            target_spacing_kpts,
                            crop_start,
                            crop_end,
                            crop_outside=True,
                        )
                        self.patch_item_metas.append(
                            {
                                "tomo_idx": row_idx,
                                "tomo_id": tomo_id,
                                "ori_spacing": ori_spacing,
                                "ori_shape": ori_shape,
                                "tomo_shape": target_spacing_tomo_shape,
                                "patch_position": patch_pos,
                                "ori_kpts": ori_kpts,
                                "num_cropped_kpts": len(_cropped_kpts),
                            }
                        )
                elif cfg.data.sampling.method == "rand_crop":
                    if len(ori_kpts) == 0:
                        rand_crop_neg_tomo_ids.extend(
                            [tomo_id] * cfg.data.sampling.rand_crop.neg_weight
                        )
                    else:
                        rand_crop_pos_tomo_ids.extend(
                            [tomo_id] * cfg.data.sampling.rand_crop.pos_weight
                        )

            assert len(self.tomo_ids) == len(self.all_tomos_meta)

            if cfg.data.sampling.method == "pre_patch":
                # foreground vs background
                fg_patches = []
                bg_patches_from_pos = []
                bg_patches_from_neg = []
                fg_num_cropped_kpts = np.array(
                    [
                        e["num_cropped_kpts"]
                        for e in self.patch_item_metas
                        if e["num_cropped_kpts"] > 0
                    ]
                )
                logger.info(
                    "OVERSAMPLE DATASET number of keypoints per patch: min=%f mean=%f median=%f max=%f",
                    np.min(fg_num_cropped_kpts),
                    np.mean(fg_num_cropped_kpts),
                    np.median(fg_num_cropped_kpts),
                    np.max(fg_num_cropped_kpts),
                )
                avg_fg_num_cropped_kpts = np.mean(fg_num_cropped_kpts)
                for patch_meta in self.patch_item_metas:
                    num_cropped_kpts = patch_meta["num_cropped_kpts"]
                    if num_cropped_kpts > 0:
                        # minimum oversample weight is 1
                        oversample_weight = max(
                            1, round(num_cropped_kpts / avg_fg_num_cropped_kpts)
                        )
                        oversample_weight = min(
                            cfg.data.sampling.pre_patch.fg_max_dup, oversample_weight
                        )
                        fg_patches.extend(
                            [patch_meta for _ in range(oversample_weight)]
                        )
                    else:
                        if len(patch_meta["ori_kpts"]) == 0:
                            bg_patches_from_neg.append(patch_meta)
                        else:
                            bg_patches_from_pos.append(patch_meta)
                bg_patches = bg_patches_from_neg + bg_patches_from_pos
                # oversampling the negative patches came from negative tomogram
                # for each neg sample, give weight 1.0 for patch came from negative tomo
                # now, we compute the weight for each patch came from positive tomo
                bg_from_pos_weight = (
                    len(bg_patches_from_neg)
                    * cfg.data.sampling.pre_patch.bg_from_pos_ratio
                    / len(bg_patches_from_pos)
                )
                if bg_from_pos_weight == 0:
                    assert len(bg_patches_from_neg) == 0
                    bg_from_pos_weight = 1
                bg_weights = [1.0] * len(bg_patches_from_neg) + [
                    bg_from_pos_weight
                ] * len(bg_patches_from_pos)
                self.fg_patches = fg_patches
                self.bg_patches = bg_patches
                self.bg_weights = bg_weights
                self.num_fg_patches_per_epoch = len(fg_patches) * len(
                    self.tta_transforms
                )
                self.num_bg_patches_per_epoch = round(
                    cfg.data.sampling.pre_patch.bg_ratio * self.num_fg_patches_per_epoch
                )
                logger.info(
                    "PRE-PATCH DATASET: loaded %s dataset with %d foreground patches (%d * %d) and %d background patches (total %d neg + %d pos = %d total, bg_from_pos_weight=%f)",
                    stage,
                    self.num_fg_patches_per_epoch,
                    len(fg_patches),
                    self.num_ttas,
                    self.num_bg_patches_per_epoch,
                    len(bg_patches_from_neg),
                    len(bg_patches_from_pos),
                    len(self.bg_patches),
                    bg_from_pos_weight,
                )
            elif cfg.data.sampling.method == "rand_crop":
                self.rand_crop_tomo_ids = (
                    rand_crop_pos_tomo_ids + rand_crop_neg_tomo_ids
                )
                logger.info(
                    "RANDOM-CROP DATASET: loaded %s dataset with len %d (%d * %d), %d positive, %d negative (active %d/%d tomo_ids)",
                    stage,
                    len(self),
                    len(self.rand_crop_tomo_ids),
                    self.num_ttas,
                    len(rand_crop_pos_tomo_ids),
                    len(rand_crop_neg_tomo_ids),
                    len(set(self.rand_crop_tomo_ids)),
                    len(self.tomo_ids),
                )
            else:
                raise ValueError

        # Setup Transform/Augmentation
        label_keys = []
        if stage == "train":
            self.transform = build_train_transform(
                cfg, image_keys=["image"], label_keys=label_keys, log_stats=False
            )
        elif cfg.data.io_backend != "cache":
            self.transform = build_val_transform(
                cfg, image_keys=["image"], label_keys=label_keys, log_stats=False
            )
        else:
            self.transform = None

        if self.transform is not None:
            self._monai_transforms = [self.transform]
        else:
            self._monai_transforms = []

        # MIXER: MIXUP, CUTMIX
        def _rand_train_item_generator():
            while True:
                idx = random.randrange(0, len(self))
                data = self._get_train_item(idx, mix=False)
                yield data["image"], data["heatmap"]

        if stage == "train":
            image_target_generator = _rand_train_item_generator()
            self.mixer_transform = T.OneOf(
                [
                    CT.MixUpFromGenerator(
                        image_target_generator,
                        prob=cfg.data.aug.mixup_prob + cfg.data.aug.cutmix_prob,
                        alpha=cfg.data.aug.mixer_alpha,
                        target_mode=cfg.data.aug.mixup_target_mode,
                    ),
                    CT.CutmixFromGenerator(
                        image_target_generator,
                        prob=cfg.data.aug.mixup_prob + cfg.data.aug.cutmix_prob,
                        alpha=cfg.data.aug.mixer_alpha,
                        inplace=True,
                    ),
                ],
                weights=(
                    [cfg.data.aug.mixup_prob, cfg.data.aug.cutmix_prob]
                    if cfg.data.aug.mixup_prob + cfg.data.aug.cutmix_prob > 0
                    else [1.0, 1.0]
                ),
                map_items=False,
                unpack_items=True,
            )
            self._monai_transforms.append(self.mixer_transform)

        logger.info(
            "%s transform:\n%s", stage, pprint.pformat(self.transform, indent=4)
        )

        # Start fast prefetch eval dataloader process
        self.shm_prefix = str(time.time()).replace(".", "")
        if stage != "train":
            self.fast_val_tomo_loader = FastValTomoLoader(
                self.all_tomos_meta,
                data_dir=self.data_dir,
                io_backend=self.io_backend,
                target_spacing=self.target_spacing,
                transform=self.transform,
                prefetch=cfg.data.fast_val_prefetch,
                # keep = (loader.prefetch_factor * loader.val_batch_size + 1) / (num_ttas * min_num_patches_per_tomo)
                # but this simple heuristic work fine
                keep=cfg.data.fast_val_prefetch + 3,
                interpolation_mode=cfg.data.transform.resample_mode,
                shm_prefix=self.shm_prefix,
                num_workers=cfg.data.fast_val_workers,
            )

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
            if self.cfg.data.sampling.method == "pre_patch":
                return self.num_fg_patches_per_epoch + self.num_bg_patches_per_epoch
            elif self.cfg.data.sampling.method == "rand_crop":
                return len(self.rand_crop_tomo_ids) * self.num_ttas
            else:
                raise ValueError
        else:
            return len(self.patch_item_metas)

    def __del__(self):
        logger.warning(
            "%s.__del__() triggered, attempt clean up..", self.__class__.__name__
        )
        if getattr(self, "tomo_loader", None) is not None:
            if hasattr(self.tomo_reader, "close"):
                self.tomo_reader.close()

    @property
    def worker_init_fn(self):
        def _woker_init_fn(worker_id: int, rank=None):
            worker_info = torch.utils.data.get_worker_info()
            dataset: Heatmap3dDataset = worker_info.dataset
            logger.debug(
                "Worker %d/%d: set MONAI transform random state to %d",
                worker_info.id + 1,
                worker_info.num_workers,
                worker_info.seed,
            )
            assert id(dataset) == id(self)
            for tf in dataset._monai_transforms:
                tf.set_random_state(seed=worker_info.seed)

            # for val, set pytorch's number of threads to 1
            # otherwise, it's stuck..
            # ref: https://github.com/pytorch/pytorch/issues/89693
            if self.stage != "train":
                # print('{self.stage} OLD NUM THREADS:', torch.get_num_threads())
                torch.set_num_threads(1)

        return _woker_init_fn

    # @lru_cache(maxsize=1)
    def read_tomo(self, tomo_id):
        if self.io_backend == "cv2_seq" and self.tomo_reader is None:
            self.tomo_reader = OpencvTomogramLoader()
        elif self.io_backend == "cv2" and self.tomo_reader is None:
            # multithread code should go after fork, so not impl in __init__()
            self.tomo_reader = MultithreadOpencvTomogramLoader(num_workers=8)
        return read_tomo(tomo_id, self.data_dir, self.io_backend, self.tomo_reader)

    def _get_val_item(self, idx):
        patch_item_meta = self.patch_item_metas[idx]
        tomo_id = patch_item_meta["tomo_id"]
        ori_spacing = patch_item_meta["ori_spacing"]
        ori_shape = patch_item_meta["ori_shape"]  # (Z, Y, X)
        tomo_shape = patch_item_meta["tomo_shape"]  # (Z, Y, X)
        patch_position = patch_item_meta["patch_position"]

        # the target-spaced tomo, not the original one
        tomo, kpts, _, __ = self.fast_val_tomo_loader.load(tomo_id)
        assert tuple(tomo.shape) == tomo_shape and tomo.dtype == torch.uint8
        logger.debug(
            "Receive item idx=%d tomo_id=%s tomo=%s kpts=%s",
            idx,
            tomo_id,
            tomo.shape,
            kpts.shape,
        )

        _roi_start, _roi_end, patch_start, patch_end = patch_position
        top_pad_z, top_pad_y, top_pad_x = [max(-start, 0) for start in patch_start]
        bot_pad_z, bot_pad_y, bot_pad_x = [
            max(0, end - size) for end, size in zip(patch_end, tomo_shape)
        ]
        actual_crop_start = [max(0, start) for start in patch_start]
        actual_crop_end = [min(end, size) for end, size in zip(patch_end, tomo_shape)]
        crop_slices = tuple(
            slice(start, end) for start, end in zip(actual_crop_start, actual_crop_end)
        )
        crop = tomo[crop_slices]

        # crop keypoints
        kpts = kpts_spatial_crop(
            kpts,
            patch_start,
            patch_end,
            crop_outside=self.cfg.data.crop_outside,
        )

        # pad if needed
        pad = (top_pad_x, bot_pad_x, top_pad_y, bot_pad_y, top_pad_z, bot_pad_z)
        if any(pad):
            crop = F.pad(crop, pad, mode="constant", value=0)
        assert crop.shape == self.cfg.data.patch_size

        if self.cfg.data.transform.heatmap_same_sigma:
            sigma = torch.tensor(
                [[self.sigma * 1000.0 / spacing for spacing in self.target_spacing]]
            )  # 1x3
            # (x, y, z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz, class)
            kpts[:, 3:6] = sigma**2
            kpts[:, 6:9] = 0.0
        heatmap = generate_heatmap(
            1,
            crop.shape,
            kpts,
            stride=self.heatmap_stride,
            heatmap_mode=self.heatmap_mode,
            lower=self.cfg.data.label_smooth[0],
            upper=self.cfg.data.label_smooth[1],
            same_std=self.cfg.data.transform.heatmap_same_std,
            dtype=torch.float16,
        )

        # apply TTA
        # @TODO - TTA keypoints as well
        # WARNING: keypoints is not TTA-transformed, result in wrong keypoints relative to transformed image
        # We disable keypoints features temporarily
        tta_idx = patch_item_meta["tta_idx"]
        _tta_name, tta_tf = self.tta_transforms[tta_idx]
        crop, _weight = tta_tf.transform(crop)
        heatmap, _weight = tta_tf.transform(heatmap)
        assert crop.dtype == torch.uint8

        return {
            "idx": torch.tensor(idx, dtype=torch.int64),
            "image": crop[None],
            # "num_kpts": torch.tensor(kpts.shape[0], dtype=torch.int64),
            # "kpts": pad_kpts(kpts, self.MAX_NUM_KPTS),
            "heatmap": heatmap,
            "tomo_idx": torch.tensor(patch_item_meta["tomo_idx"], dtype=torch.int64),
            "ori_spacing": torch.tensor(ori_spacing, dtype=torch.float32),
            "target_spacing": torch.tensor(self.target_spacing, dtype=torch.float32),
            "patch_position": patch_position,
            "ori_shape": torch.tensor(ori_shape, dtype=torch.int16),
            "tomo_shape": torch.tensor(tomo_shape, dtype=torch.int16),  # spaced
            "tta_idx": torch.tensor(tta_idx, dtype=torch.int16),
            "patch_is_first": torch.tensor(
                patch_item_meta["is_first"], dtype=torch.bool
            ),
            "patch_is_last": torch.tensor(patch_item_meta["is_last"], dtype=torch.bool),
        }

    def _get_train_item(self, idx, mix=False):
        patch_type = "unk"
        if self.cfg.data.sampling.method == "pre_patch":
            # Oversample foreground patches
            if idx >= self.num_fg_patches_per_epoch:
                # sample bg patch
                patch_type = "bg"
                patch_meta = random.choices(
                    self.bg_patches, weights=self.bg_weights, k=1
                )[0]
                tta_idx, tta_tf = random.choice(self.tta_transforms)
            else:
                # sample fg patch
                patch_type = "fg"
                fg_patch_idx = idx // self.num_ttas
                tta_idx = idx % self.num_ttas
                patch_meta = self.fg_patches[fg_patch_idx]
                tta_name, tta_tf = self.tta_transforms[tta_idx]
            tomo_id = patch_meta["tomo_id"]
            tomo_shape = patch_meta["tomo_shape"]
            ori_spacing = patch_meta["ori_spacing"]
            ori_kpts = patch_meta["ori_kpts"]
        elif self.cfg.data.sampling.method == "rand_crop":
            # normal Random Crop training
            tomo_idx = idx // self.num_ttas
            tta_idx = idx % self.num_ttas
            tomo_id = self.rand_crop_tomo_ids[tomo_idx]
            tomo_meta = self.all_tomos_meta[tomo_id]
            ori_spacing = tomo_meta["ori_spacing"]
            tomo_shape = tomo_meta["tomo_shape"]
            ori_kpts = tomo_meta["ori_kpts"]
            tta_name, tta_tf = self.tta_transforms[tta_idx]
        else:
            raise ValueError

        if self.io_backend == "cache":
            # memmap, Tensor
            tomo, kpts = load_spaced_val_sample_cache(
                tomo_id,
                tomo_shape,
                self.data_dir,
                ori_kpts,
                ori_spacing,
                self.target_spacing,
                self.cfg.data.transform.resample_mode,
                return_type="mmap",
            )
        else:
            # Tensor, Tensor
            tomo = self.read_tomo(tomo_id)
            kpts = ori_kpts

        if self.cfg.data.sampling.method == "pre_patch":
            _roi_start, _roi_end, patch_start, patch_end = patch_meta["patch_position"]
            top_pad_z, top_pad_y, top_pad_x = [max(-start, 0) for start in patch_start]
            bot_pad_z, bot_pad_y, bot_pad_x = [
                max(0, end - size) for end, size in zip(patch_end, tomo_shape)
            ]
            actual_crop_start = [max(0, start) for start in patch_start]
            actual_crop_end = [
                min(end, size) for end, size in zip(patch_end, tomo_shape)
            ]
            crop_slices = tuple(
                slice(start, end)
                for start, end in zip(actual_crop_start, actual_crop_end)
            )
            if self.io_backend == "cache":
                # just crop and load the patch only, using numpy memmap
                crop = torch.from_numpy(tomo[crop_slices])  # ZYX
                assert crop.dtype == torch.uint8
                # crop keypoints
                kpts = kpts_spatial_crop(
                    kpts,
                    actual_crop_start,
                    actual_crop_end,
                    crop_outside=self.cfg.data.crop_outside,
                )  # ZYX order
                assert len(kpts.shape) == 2 and kpts.shape[1] == 10
                kpts = kpts[:, [2, 1, 0, 5, 4, 3, 8, 7, 6, 9]]  # ZYX order to XYZ order
                # dict of MONAI
                data = {
                    "image": crop.permute(2, 1, 0)[None],  # ZYX -> 1XYZ = 1HWD,
                    "kpts": kpts[None],  # (1, N, 10), XYZ order
                }
            else:
                # add `crop_slices` so that CT.CropBySlicesd() can do the lazy cropping
                crop_slices = (
                    crop_slices[2],
                    crop_slices[1],
                    crop_slices[0],
                )  # ZYX -> XYZ for compatible with MONAI Transform
                data = {
                    "image": tomo.permute(2, 1, 0)[None],  # ZYX -> 1XYZ = 1HWD
                    "spacing_scale": (
                        self.target_spacing[2] / ori_spacing[2],
                        self.target_spacing[1] / ori_spacing[1],
                        self.target_spacing[0] / ori_spacing[0],
                    ),  # HWD or YXZ
                    "kpts": kpts[None],  # (1, N, 10), XYZ order
                    "crop_slices": crop_slices,
                }
        elif self.cfg.data.sampling.method == "rand_crop":
            if self.io_backend == "cache":
                tomo = torch.from_numpy(tomo)  # ZYX
            assert isinstance(tomo, torch.Tensor)
            spacing_scale = torch.tensor(
                [
                    [
                        self.target_spacing[2] / ori_spacing[2],
                        self.target_spacing[1] / ori_spacing[1],
                        self.target_spacing[0] / ori_spacing[0],
                    ]
                ]
            )  # (1, 3), XYZ
            spaced_kpts = kpts.clone()
            spaced_kpts[:, :3] /= spacing_scale
            assert torch.all(spaced_kpts[:, 6:9] == 0)
            spaced_kpts[:, 3:6] /= spacing_scale**2
            data = {
                "image": tomo.permute(2, 1, 0)[None],  # ZYX -> 1XYZ = 1HWD
                "spacing_scale": spacing_scale,
                "kpts": kpts[None],  # (1, N, 10), XYZ order
                "spaced_kpts": spaced_kpts[None],  # (1, N, 10), XYZ order
            }
        else:
            raise ValueError

        # number of trials to get foreground patch
        RETRY = 5
        for _ in range(RETRY):
            tdata = self.transform(data, threading=False)
            crop = tdata["image"].permute(0, 3, 2, 1)  # 1XYZ -> 1ZYX
            kpts = tdata["kpts"]
            assert len(kpts.shape) == 3 and kpts.shape[0] == 1 and kpts.shape[2] == 10
            kpts = kpts[0, :, [2, 1, 0, 5, 4, 3, 8, 7, 6, 9]]  # XYZ order to ZYX order

            # filter invalid keypoints
            kpts = kpts_spatial_crop(
                kpts, [0, 0, 0], crop.shape[1:], crop_outside=self.cfg.data.crop_outside
            )
            # at least one foreground keypoint
            if patch_type == "bg" or len(kpts) > 0 or (not self.cfg.data.ensure_fg):
                break

        # pad if needed
        if self.cfg.data.sampling.method == "pre_patch":
            pad = (top_pad_x, bot_pad_x, top_pad_y, bot_pad_y, top_pad_z, bot_pad_z)
            if any(pad):
                crop = F.pad(crop, pad, mode="constant", value=0)
                kpts[:, :3] += torch.tensor([top_pad_z, top_pad_y, top_pad_x])
        else:
            # @TODO - currently, ensure no padding is needed
            # need to impl padding mechanism
            pass

        assert crop.shape[1:] == self.cfg.data.patch_size

        heatmap = generate_heatmap(
            1,
            crop.shape[1:],
            kpts,
            stride=self.heatmap_stride,
            heatmap_mode=self.heatmap_mode,
            lower=self.cfg.data.label_smooth[0],
            upper=self.cfg.data.label_smooth[1],
            same_std=self.cfg.data.transform.heatmap_same_std,
            dtype=torch.float16,
        )

        # apply fixed train TTA transform: rot90, flip,..
        crop, _weight = tta_tf.transform(crop)
        heatmap, _weight = tta_tf.transform(heatmap)

        ########## MIXER ##########
        # MIXUP
        if mix:
            crop, heatmap = self.mixer_transform((crop, heatmap))

        # use uint8 to save memory and speed up data transfer with large dimension tensor, with experimental results:
        # float32: 42-46 GB RAM, 2h11m
        # uint8: 30-34 GB RAM, 1h53m
        assert crop.dtype == torch.float32
        crop = torch.clip(crop, 0, 255, out=crop).to(torch.uint8)

        return {
            "idx": torch.tensor(idx, dtype=torch.int64),
            "image": crop,
            # "num_kpts": torch.tensor(kpts.shape[0], dtype=torch.int16),
            # "kpts": pad_kpts(kpts, self.MAX_NUM_KPTS),
            "heatmap": heatmap,
        }

    def __getitem__(self, idx):
        # print("PYTORCH NUMBER OF THREADS:", torch.get_num_threads())
        if self.stage == "train":
            return self._get_train_item(idx, mix=True)
        else:
            return self._get_val_item(idx)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    logging.basicConfig(level=logging.INFO)

    yaml_str = """

env:
    data_dir: /home/dangnh36/datasets/.comp/byu/

cv:
    strategy: skf4_rd42
    num_folds: 4
    fold_idx: 0

loader:
    num_workers: 1

data:
    patch_size: [224, 448, 448]
    start: [0, 0, 0]
    overlap: [0, 0, 0]
    border: [0, 0, 0]

    sigma: 0.2
    fast_val_workers: 5
    fast_val_prefetch: 5
    io_backend: cv2  # cv2, cv2_seq, npy, cache
    crop_outside: True
    ensure_fg: False
    label_smooth: [0.0,1.0]


    sampling:
        method: pre_patch  # pre_patch | rand_crop
        pre_patch:
            fg_max_dup: 1
            bg_ratio: 0.0
            bg_from_pos_ratio: 0.01
            overlap: [0, 0, 0]
        rand_crop:
            random_center: True
            margin: 0.25
            auto_correct_center: True
            pos_weight: 1
            neg_weight: 1

    transform:
        resample_mode: trilinear # F.grid_sample() mode
        target_spacing: [16.0, 16.0, 16.0]

        heatmap_mode: gaussian
        heatmap_stride: 1
        heatmap_same_std: False
        lazy: True
        device: null

    aug:
        enable: True
        zoom_prob: 0.4
        zoom_range: [[0.6, 1.2], [0.6, 1.2], [0.6, 1.2]]  # (X, Y, Z) or (H, W, D)
        # affine1
        affine1_prob: 0.5
        affine1_scale: 0.3  # max_skew_xy = 1.3 / 0.7 = 1.86
        # affine2
        affine2_prob: 0.25
        affine2_rotate_xy: 15 # degrees
        affine2_scale: 0.3  # max_skew_xy = 1.3 / 0.7 = 1.86
        affine2_shear: 0.2

        rand_shift: False  # only used in rand_crop, very useful if random_center=auto_correct_center=False

        # no lazy, can't properly transform points
        grid_distort_prob: 0.0
        smooth_deform_prob: 0.0

        intensity_prob: 0.5
        smooth_prob: 0.0
        hist_equalize: False
        downsample_prob: 0.2
        coarse_dropout_prob: 0.1

        # MIXER
        mixup_prob: 0.0
        cutmix_prob: 0.0
        mixer_alpha: 1.0
        mixup_target_mode: max

    tta:
        # enable: [zyx]
        enable: [zyx, zxy, zyx_x, zyx_y]
        # enable: [zyx, zyx_x, zyx_y, zyx_z, zyx_xy, zyx_xz, zyx_yz, zyx_xyz, zxy, zxy_x, zxy_y, zxy_z, zxy_xy, zxy_xz, zxy_yz, zxy_xyz]
"""
    global_cfg = OmegaConf.create(yaml_str)
    dataset = Heatmap3dDataset(global_cfg, stage="val")
    if dataset.stage != "train":
        dataset.fast_val_tomo_loader.start()

    TEST_MODE = 0

    if TEST_MODE == 0:
        print("LEN:", len(dataset))
        # random.seed(611)
        # select_idxs = random.choices(list(range(len(dataset))), k=500)
        select_idxs = list(range(200))
        for i in tqdm(select_idxs):
            sample = dataset[i]
            for k, v in sample.items():
                print(f"{k} -> {type(v)} {v.dtype} {v.shape}")
            print(sample["image"].min(), sample["image"].max())
            print(sample["heatmap"].min(), sample["heatmap"].max())
            print("----------------\n")
    elif TEST_MODE == 1:
        print("LEN:", len(dataset))
        from lightning.fabric.utilities.seed import pl_worker_init_function
        from torch.utils.data import DataLoader

        def _worker_init_fn(*args, **kwargs):
            # 1 -> 4, 2 -> 2.8, 4 -> 2.25, 8 -> 2.03, 32 -> 1.95, 64 -> 1.94
            torch.set_num_threads(8)
            # torch.set_num_interop_threads(32)
            pl_worker_init_function(*args, **kwargs)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=False,
            drop_last=False,
            worker_init_fn=_worker_init_fn,
        )
        for sample in dataloader:
            for k, v in sample.items():
                print(f"{k} -> {type(v)} {v.dtype} {v.shape}")
            print(sample["image"].min(), sample["image"].max())
            print(sample["heatmap"].min(), sample["heatmap"].max())
            print("----------------\n")
            pass
    else:
        raise ValueError
