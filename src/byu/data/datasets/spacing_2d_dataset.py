import logging
import os
import random

import albumentations as A
import cv2
import hydra
import numpy as np
import polars as pl
import sklearn
import torch
from omegaconf import OmegaConf
from yagm.data.datasets.base_dataset import BaseDataset

from byu.data.io import WRONG_QUANTILE_TOMO_IDS

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

MIN_SPACING = 6.0  # min=6.5 on train data
MAX_SPACING = 24.0  # max=19.733 on train
NEARBY_NUM = 0
NEARBY_STRIDE = 5


class Spacing2dDataset(BaseDataset):
    """
    Dataset used for voxel spacing regression. Developed on last day -> not carefully tested
    """

    def __init__(self, cfg, stage="train", cache={}):
        super().__init__(cfg, stage, cache)
        # by default, Pytorch use multiple threads to speedup CPU operation
        # if not set number of threads to 1 before `fork` caused by Dataloader with num_workers > 0
        # then boom! segmentfault, related to: https://github.com/pytorch/pytorch/issues/54752
        # each dataloader process then can change the number of threads by a call to torch.set_num_threads()
        # e.g, in worker_init_fn passed to DataLoader
        logger.info(
            "Temporarily set Pytorch number of threads from %d to 1",
            torch.get_num_threads(),
        )
        torch.set_num_threads(1)

        self.data_dir = cfg.env.data_dir

        df = pl.read_csv(os.path.join(self.data_dir, "processed", "all_gt_v3.csv"))
        all_tomo_ids = sorted(df["tomo_id"].unique().to_list())
        tomo_meta_dict = {}
        for row in df.iter_rows(named=True):
            tomo_meta_dict[row["tomo_id"]] = row
        # CV SPLIT
        splitter = sklearn.model_selection.KFold(
            n_splits=5, shuffle=True, random_state=42
        )
        for fold_idx, (train_idxs, val_idxs) in enumerate(splitter.split(all_tomo_ids)):
            if fold_idx != cfg.cv.fold_idx:
                continue
            else:
                val_tomo_ids = [all_tomo_ids[idx] for idx in val_idxs]
                train_tomo_ids = [all_tomo_ids[idx] for idx in train_idxs]
                break
        self.tomo_ids = train_tomo_ids if stage == "train" else val_tomo_ids

        # BUILD TRANSFORM FUNC
        if stage == "train":
            NUM_BINS = 20
            spacing_bins = np.linspace(MIN_SPACING, MAX_SPACING, NUM_BINS + 1).tolist()
            logger.info("TRAIN SPACING BINS: %s", spacing_bins)
            train_samples = []
            for tomo_id in self.tomo_ids:
                for bin_idx in range(NUM_BINS):
                    train_samples.append(
                        {
                            "tomo_id": tomo_id,
                            "ori_spacing": tomo_meta_dict[tomo_id]["voxel_spacing"],
                            "target_spacing_bin": (
                                spacing_bins[bin_idx],
                                spacing_bins[bin_idx + 1],
                            ),
                            "Z": tomo_meta_dict[tomo_id]["Z"],
                        }
                    )
            self.samples = train_samples
        else:
            val_samples = []
            NUM_TARGET_SPACING_PER_TOMO = 8
            NUM_SLICES_PER_TOMO = 3
            target_spacings = np.linspace(
                MIN_SPACING, MAX_SPACING, NUM_TARGET_SPACING_PER_TOMO
            ).tolist()
            for tomo_id in self.tomo_ids:
                Z = tomo_meta_dict[tomo_id]["Z"]
                select_z_list = np.linspace(0, Z, NUM_SLICES_PER_TOMO + 2)[1:-1]
                for select_z in select_z_list:
                    for target_spacing in target_spacings:
                        val_samples.append(
                            {
                                "tomo_id": tomo_id,
                                "ori_spacing": tomo_meta_dict[tomo_id]["voxel_spacing"],
                                "target_spacing": target_spacing,
                                "select_z": select_z,
                                "Z": Z,
                            }
                        )

            self.samples = val_samples

        # augmentation & transform
        self.transform = self.build_augmentation_transform()
        logger.info("%s transform:\n%s", self.stage, self.transform)

    def __len__(self):
        return len(self.samples)

    def _load_imgs(self, tomo_id, select_z, Z, ori_spacing, target_spacing):
        if NEARBY_NUM > 0:
            assert NEARBY_NUM % 2 == 0
            half = NEARBY_NUM // 2
            select_zs = [select_z + NEARBY_STRIDE * i for i in range(-half, half + 1)]
            select_zs = [round(min(max(0, e), Z - 1)) for e in select_zs]
            assert len(select_zs) == NEARBY_NUM + 1
        else:
            select_zs = [round(select_z)]

        imgs = []
        if tomo_id.startswith("tomo_"):
            tomo_dir = os.path.join(self.data_dir, "raw", "train", tomo_id)
        else:
            tomo_dir = os.path.join(self.data_dir, "external", "tomogram", tomo_id)
        interpolation_mode = random.choice(
            [
                cv2.INTER_LINEAR,
                cv2.INTER_AREA,
                cv2.INTER_NEAREST,
                cv2.INTER_LANCZOS4,
                cv2.INTER_CUBIC,
                cv2.INTER_LINEAR_EXACT,
            ]
        )
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
                fx=ori_spacing / target_spacing,
                fy=ori_spacing / target_spacing,
                interpolation=interpolation_mode,
            )
            imgs.append(img)
        return imgs

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tomo_id = sample["tomo_id"]
        ori_spacing = sample["ori_spacing"]
        Z = sample["Z"]
        if self.stage == "train":
            target_spacing = random.uniform(*sample["target_spacing_bin"])
            select_z = random.randrange(0, Z)
        else:
            target_spacing = sample["target_spacing"]
            select_z = sample["select_z"]
        imgs = self._load_imgs(tomo_id, select_z, Z, ori_spacing, target_spacing)
        img, data = self.do_albumentations_transform(imgs, {})
        assert img.ndim == 3 and img.shape[1:] == self.cfg.data.patch_size
        assert MIN_SPACING <= target_spacing <= MAX_SPACING, f"{target_spacing}"
        target = (target_spacing - MIN_SPACING) / (MAX_SPACING - MIN_SPACING)
        return {
            "idx": torch.tensor(idx, dtype=torch.long),
            "image": torch.from_numpy(img),  # (1, H, W)
            "target_spacing": torch.tensor(target_spacing, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }

    def build_augmentation_transform(self):
        # additional_targets = {
        #     f"image{i}": "image" for i in range(NUM_NEARBY - 1)
        # }
        additional_targets = {}
        if self.stage != "train":
            transform = A.Compose(
                [
                    A.PadIfNeeded(
                        self.cfg.data.patch_size[0],
                        self.cfg.data.patch_size[1],
                        position="top_left",
                        border_mode=cv2.BORDER_CONSTANT,
                        value=128,
                        p=1.0,
                    ),
                    A.CenterCrop(
                        self.cfg.data.patch_size[0], self.cfg.data.patch_size[1], p=1.0
                    ),
                ],
                additional_targets=additional_targets,
                p=1.0,
            )
            return transform

        # if in train stage
        aug = A.Compose(
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
                A.Affine(
                    scale=None,
                    translate_percent=None,
                    rotate=(0, 360),
                    shear=None,
                    interpolation=cv2.INTER_LINEAR,
                    cval=128,
                    mode=cv2.BORDER_CONSTANT,
                    fit_output=False,
                    keep_ratio=False,
                    balanced_scale=False,
                    p=0.5,
                ),
                A.PadIfNeeded(
                    self.cfg.data.patch_size[0],
                    self.cfg.data.patch_size[1],
                    position="top_left",
                    border_mode=cv2.BORDER_CONSTANT,
                    value=128,
                    p=1.0,
                ),
                A.RandomCrop(
                    self.cfg.data.patch_size[0], self.cfg.data.patch_size[1], p=1.0
                ),
            ],
            additional_targets=additional_targets,
            p=1.0,
        )
        return aug

    def do_albumentations_transform(self, imgs, data):
        num_imgs = len(imgs)
        assert num_imgs == 1
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from omegaconf import OmegaConf

    yaml_str = """

cv:
    strategy: unk
    num_folds: 4
    fold_idx: 0

env:
    data_dir: '/home/dangnh36/datasets/.comp/byu/'

data:
    _target_: yagm.data.base_datamodule.BaseDataModule
    _dataset_target_: byu.data.datasets.spacing_2d_dataset.Spacing2dDataset

    patch_size: [512, 512]
"""
    from lightning import seed_everything

    STAGE = "train"
    seed_everything(42)
    global_cfg = OmegaConf.create(yaml_str)
    dataset = Spacing2dDataset(cfg=global_cfg, stage=STAGE)
    print("DATASET LEN:", len(dataset))

    for idx in range(0, len(dataset), 1000):
        sample = dataset[idx]
        print(idx, {k: [v.shape, v.dtype] for k, v in sample.items()})
