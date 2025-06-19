from typing import List, Tuple

import numpy as np
import torch
from yagm.data.datasets.base_dataset import BaseDataset
from yagm.transforms.keypoints import encode as kpts_encode


def generate_heatmap(
    num_classes: int,
    image_shape: Tuple[int, int, int],
    kpts: torch.Tensor,
    stride: Tuple[int, int, int],
    heatmap_mode: str = "gaussian",
    lower=0.0,
    upper=1.0,
    same_std=False,
    dtype=torch.float32,
):
    assert heatmap_mode in ["gaussian", "segment", "point"]
    assert len(kpts) == 0 or kpts.shape[-1] == 10
    if heatmap_mode == "gaussian":
        heatmap = kpts_encode.generate_3d_gaussian_heatmap(
            (num_classes, *image_shape),
            kpts,
            stride=stride,
            covariance=None,
            dtype=dtype,
            sigma_scale_factor=None,
            conf_interval=0.999,
            lower=lower,
            upper=upper,
            same_std=same_std,
            add_offset=True,
            validate_cov_mat=False,
        )
    elif heatmap_mode == "segment":
        heatmap = kpts_encode.generate_3d_segment_mask(
            (num_classes, *image_shape),
            kpts,
            stride=stride,
            covariance=None,
            dtype=dtype,
            sigma_scale_factor=None,
            conf_interval=0.999,
            lower=lower,
            upper=upper,
            same_std=same_std,
            add_offset=True,
            validate_cov_mat=False,
        )
    elif heatmap_mode == "point":
        heatmap = kpts_encode.generate_3d_point_mask(
            (num_classes, *image_shape),
            kpts,
            stride=stride,
            dtype=dtype,
            lower=lower,
            upper=upper,
            add_offset=True,
        )
    else:
        raise ValueError
    return heatmap


def compute_target_spacing_shape(
    ori_shape: list | tuple,
    ori_spacing: list | tuple,
    target_spacing: list | tuple,
    method="torch",
    scale_extent=False,
):
    assert len(ori_shape) == len(ori_spacing) == len(target_spacing)
    if method == "monai":
        # ref: https://github.com/Project-MONAI/MONAI/blob/dev/monai/data/utils.py#L875
        in_coords = np.array(
            [
                (-0.5, dim - 0.5) if scale_extent else (0.0, dim - 1.0)
                for dim in ori_shape
            ]
        )  # (ndim, 2)
        ori_spacing_np = np.array(ori_spacing)[..., None]  # (ndim, 1)
        target_spacing_np = np.array(target_spacing)[..., None]  # (ndim, 1)
        out_shape = np.ptp(in_coords * ori_spacing_np / target_spacing_np, axis=1)
        out_shape = np.round(out_shape) if scale_extent else np.round(out_shape + 1.0)
        out_shape = tuple(int(e) for e in out_shape)
        return out_shape
    elif method == "torch":
        return tuple(
            int(e * os / ts)
            for e, os, ts in zip(ori_shape, ori_spacing, target_spacing)
        )
    else:
        raise ValueError
