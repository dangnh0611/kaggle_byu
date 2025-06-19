import logging
import os
from typing import Tuple

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from IPython.display import Image
from IPython.display import display as idisplay

logger = logging.getLogger(__name__)


def longest_resize(
    img, max_h=None, max_w=None, upscale=False, interpolation=cv2.INTER_LINEAR
):
    if max_h is None and max_w is None:
        return img
    img_h, img_w = img.shape[:2]
    _ratios = []
    if max_h is not None:
        _ratios.append(max_h / img_h)
    if max_w is not None:
        _ratios.append(max_w / img_w)
    r = min(_ratios)
    if not upscale:
        r = min(1.0, r)
    if r == 1.0:
        return img
    new_h, new_w = int(r * img_h), int(r * img_w)
    img = cv2.resize(img, (new_w, new_h), interpolation)
    return img


def display_img(img, max_h=None, max_w=None):
    img = longest_resize(img, max_h, max_w)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    idisplay(PIL.Image.fromarray(img))


def create_heatmap_image(data: np.ndarray) -> np.ndarray:
    """
    Converts a 2D NumPy array of floats in [0, 1] to a heatmap image (uint8 RGB).

    Args:
        data: 2D NumPy array with values in [0, 1].

    Returns:
        heatmap: 3D NumPy array of uint8 RGB values (H, W, 3).
    """
    _min = data.min()
    _max = data.max()
    assert (
        _min >= 0 and _max <= 1
    ), f"Data must be in the range [0, 1], got min={_min} max={_max}"

    # Get colormap (returns RGBA floats in [0, 1])
    colormap = plt.get_cmap("viridis")
    colored_data = colormap(data)  # shape: (H, W, 4)

    # Convert to uint8 RGB
    heatmap = (colored_data[:, :, :3] * 255).astype(np.uint8)
    return heatmap


def viz_img_heatmap_pair(
    tomo_dir: str,
    ori_spacing: float,
    ori_shape: Tuple[int],
    heatmap: torch.Tensor,
    ref_keypoint: list[list[float]],
    sub_keypoints: list[list[float]] | None = None,
    tag: str = "",
    ref_color: Tuple[int] = (255, 0, 0),
    sub_color: Tuple[int] = (0, 255, 0),
    draw_title: bool = False,
    include_neighbor=False,
):
    """
    Returns: RGB visualization image
    """
    Z, Y, X = [int(e) for e in ori_shape]
    Z2, Y2, X2 = heatmap.shape
    r = round(1000 / ori_spacing)
    z, y, x = [round(e) for e in ref_keypoint[:3]]
    if len(ref_keypoint) == 3:
        conf = None
    else:
        conf = round(ref_keypoint[3] * 100)

    if z == -1:
        has_kpt = False
        assert z == y == x == -1
        z = Z // 2
    else:
        has_kpt = True
    if z > Z:
        raise ValueError(f"Invalid value of z={z} Z={Z}")
    elif z == Z:
        logger.warning("Visualization with z==Z==%d", z)
        z -= 1

    img_viz = cv2.imread(os.path.join(tomo_dir, f"slice_{z:04d}.jpg"), cv2.IMREAD_COLOR)
    Y, X = img_viz.shape[:2]
    heatmap_viz = create_heatmap_image(
        heatmap[min(round(z * Z2 / Z), Z2 - 1)].cpu().numpy()
    )
    heatmap_viz = cv2.resize(heatmap_viz, (X, Y))

    if draw_title:
        cv2.putText(
            heatmap_viz,
            f"ORI: {Z} {Y} {X}",
            (0, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            thickness=2,
        )
        cv2.putText(
            heatmap_viz,
            f"SPA: {Z2} {Y2} {X2}",
            (0, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            thickness=2,
        )
        cv2.putText(
            heatmap_viz,
            f"R:{r}",
            (0, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            thickness=2,
        )

    if sub_keypoints is not None:
        for sub_kpt in sub_keypoints:
            z2, y2, x2 = [round(e) for e in sub_kpt[:3]]
            r2 = round(max(0.0, (r**2 - (z2 - z) ** 2)) ** 0.5)
            assert r2 >= 0
            if r2 > 0:
                cv2.circle(img_viz, (x2, y2), radius=r2, color=sub_color, thickness=3)
                cv2.circle(
                    heatmap_viz, (x2, y2), radius=r2, color=sub_color, thickness=3
                )

    if has_kpt:
        cv2.circle(img_viz, (x, y), radius=r, color=ref_color, thickness=3)
        cv2.circle(heatmap_viz, (x, y), radius=r, color=ref_color, thickness=3)
        cv2.putText(
            heatmap_viz,
            f'{tag}: {z} {y} {x} {conf if conf is not None else ""}',
            (0, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            thickness=2,
        )

    if not include_neighbor:
        final_viz = np.concatenate([img_viz, heatmap_viz], axis=0)  # (2H, W, C)
    else:
        img_viz_before = cv2.imread(
            os.path.join(tomo_dir, f"slice_{max(z-8, 0):04d}.jpg"), cv2.IMREAD_COLOR
        )
        img_viz_after = cv2.imread(
            os.path.join(tomo_dir, f"slice_{min(z+8, Z-1):04d}.jpg"), cv2.IMREAD_COLOR
        )
        final_viz = np.concatenate(
            [img_viz_before, img_viz_after, img_viz, heatmap_viz], axis=0
        )  # (4H, W, C)
    return final_viz


def viz_byu(
    tomo_dir: str,
    ori_spacing: float,
    ori_shape: Tuple[int],
    gt_zyx: list[float],
    pred_heatmap: torch.Tensor,
    pred_keypoints: list[list[float]],
    include_neighbor=False,
):
    """
    Returns: RGB visualization image
    """
    gt_viz = viz_img_heatmap_pair(
        tomo_dir,
        ori_spacing,
        ori_shape,
        pred_heatmap,
        gt_zyx,
        pred_keypoints,
        tag="GT",
        ref_color=(255, 0, 0),
        sub_color=(0, 255, 0),
        draw_title=False,
        include_neighbor=include_neighbor,
    )
    all_vizs = [gt_viz]

    for pred_idx, pred_kpt in enumerate(pred_keypoints):
        pred_viz = viz_img_heatmap_pair(
            tomo_dir,
            ori_spacing,
            ori_shape,
            pred_heatmap,
            pred_kpt,
            [gt_zyx],
            tag=f"PRED {pred_idx}",
            ref_color=(0, 255, 0),
            sub_color=(255, 0, 0),
            include_neighbor=include_neighbor,
        )
        all_vizs.append(pred_viz)
    final_viz = np.concatenate(all_vizs, axis=1)
    return final_viz


def viz_fuse_image_and_heatmap(image, heatmap, cmap_name="viridis"):
    """
    Fuse image with heatmap using pixel-wise alpha from heatmap values and a matplotlib colormap.

    Parameters:
        image (np.ndarray): Original image (H, W, 3) in BGR format.
        heatmap (np.ndarray): Heatmap (H, W) or (H, W, 1) in float [0, 1]
        cmap_name (str): Matplotlib colormap name (e.g., 'viridis', 'plasma', 'magma', 'inferno').

    Returns:
        fused (np.ndarray): Blended image in BGR format.
    """

    if heatmap.ndim == 3 and heatmap.shape[2] == 1:
        heatmap = heatmap.squeeze(-1)

    # Apply matplotlib colormap
    cmap = cm.get_cmap(cmap_name)
    heatmap_color = cmap(heatmap)[
        :, :, :3
    ]  # Drop alpha channel → (H, W, 3), RGB in [0,1]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_RGB2BGR)

    # Resize if needed
    if heatmap_color.shape[:2] != image.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Expand alpha to (H, W, 1)
    alpha = heatmap[..., np.newaxis]

    # Convert to float32
    image_float = image.astype(np.float32)
    heatmap_color_float = heatmap_color.astype(np.float32)

    # Blend pixel-by-pixel
    fused = (1.0 - alpha) * image_float + alpha * heatmap_color_float
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    return fused
