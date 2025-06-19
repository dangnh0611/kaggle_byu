import logging
from typing import Tuple
import torch
from torch.nn import functional as F


logger = logging.getLogger(__name__)


def kpts_pad_for_batching(kpts: torch.Tensor, maxlen) -> torch.Tensor:
    """
    Pad keypoints to same length for batching purpose.

    Args:
        kpts: (N, D) where N is the number of keypoints, D is keypoint dimension

    Returns:
        Padded Tensor
    """
    assert kpts.shape[0] <= maxlen
    pad = [0, 0] * kpts.ndim
    pad[-1] = maxlen - kpts.shape[0]  # bottom padding on first dim
    return F.pad(kpts, pad)


def kpts_spatial_crop(
    kpts: torch.Tensor,
    crop_start: Tuple[int, int, int],
    crop_end: Tuple[int, int, int],
    crop_outside=True,
    ndim=3,
    end_offset=-1.0,
) -> torch.Tensor:
    """
    Crop keypoints: remove out-of-regions ones, then offset valid one due to cropping

    Args:
        kpts: (N, D), where typically D>=ndim, the first ndim dimensions are coordinates.
            For example, when ndim=3, the first 3 channels indicate Z, Y, X coordinates
        crop_start: starting coordinate of length `ndim`, for example (Z, Y, X) if ndim==3
        crop_end: ending coordinate of length `ndim`, for example (Z, Y, X) if ndim==3
        crop_outside: whether to crop keypoints whose centers are outside of image region
        ndim: Number of coordinate dimension, 3 for 3D data, 2 or 2D data
        end_offset: should be -1 if in voxel indices (center of first top-left voxel has coordinate (0, 0, 0))
            or -0.5 for floating point coordinate (center of first voxel has coordinate (0.5, 0.5, 0.5))

    Returns:
        Cropped keypoints with length <= input keypoints
    """
    crop_start = torch.tensor(crop_start)[None]
    crop_end = torch.tensor(crop_end)[None] + end_offset
    assert crop_start.shape == crop_end.shape == (1, ndim)
    assert kpts.ndim == 2 and kpts.shape[1] >= ndim
    if crop_outside:
        keep = torch.all(
            torch.logical_and(kpts[:, :ndim] >= crop_start, kpts[:, :ndim] <= crop_end),
            dim=1,
        )
        kpts = kpts[keep]
    kpts[:, :ndim] = kpts[:, :ndim] - crop_start
    return kpts
