import logging
import time
from typing import Callable, Tuple

import torch
from torch.nn import functional as F
from yagm.transforms.keypoints.nms import keypoints_nms

try:
    import cc3d
except:
    print('Could not found cc3d')

logger = logging.getLogger(__name__)


@torch.inference_mode()
def decode_heatmap_3d(
    heatmap: torch.Tensor,
    pool_ksize: Tuple[int, int, int],
    nms_radius_thres: float,
    blur_operator: Callable | None = None,
    conf_thres=0.01,
    max_dets=None,
    timeout=None,
) -> torch.Tensor:
    """
    Decode 3D probability heatmap of shape (X, Y, Z)

    Args:
        radius_thres: L2 NMS threshold
        pool_ksize: using large ksize is slow due to Pytorch's implementation

    Returns:
        Tensor of shape (N, 4) where 4 means [x, y, z, conf]
    """
    _t1 = time.time()
    if blur_operator:
        # start = time.time()
        # torch.cuda.synchronize()
        heatmap = blur_operator(heatmap[None, None])[0, 0]
        # end = time.time()
        # torch.cuda.synchronize()
        # print('\nBLUR', heatmap.shape, heatmap.dtype, round((end - start) * 1000, 2), 'ms\n')
    if max_dets == 1:
        # simply take the argmax
        max_index = torch.argmax(heatmap)  # Get the flattened index of max value
        z, y, x = torch.unravel_index(max_index, heatmap.shape)  # Convert to 3D index
        conf = heatmap[z, y, x]
        if conf >= conf_thres:
            return torch.tensor([[z, y, x, conf]])
        else:
            return torch.zeros((0, 4), dtype=torch.float32)

    ori_heatmap_dtype = heatmap.dtype
    if pool_ksize and not all([e % 2 == 1 for e in pool_ksize]):
        raise ValueError("Pooling kernel size must be ood integer.")
    padding = [(k - 1) // 2 for k in pool_ksize]
    maximum = F.max_pool3d(heatmap[None], pool_ksize, stride=1, padding=padding)[0]
    heatmap = heatmap * (heatmap == maximum) * (heatmap >= conf_thres)
    assert heatmap.dtype == ori_heatmap_dtype
    keypoints = list(torch.nonzero(heatmap, as_tuple=True))
    confs = heatmap[keypoints]
    keypoints = torch.stack(keypoints + [confs], dim=-1)

    if nms_radius_thres > 0:
        _num_before_nms = keypoints.shape[0]
        _t0 = time.time()
        keep = keypoints_nms(
            keypoints, nms_radius_thres, max_dets=max_dets, timeout=timeout
        )
        _t1 = time.time()
        keypoints = keypoints[keep]
        _num_after_nms = keypoints.shape[0]
        logger.debug(
            "NMS take %s sec, %d -> %d", _t1 - _t0, _num_before_nms, _num_after_nms
        )
    return keypoints


def decode_segment_mask_3d(
    heatmap: torch.Tensor,
    conf_thres: float,
    volume_thres: float,
    max_dets: int = 1,
    conf_mode: str = "fuse",
    prob_mode: str = "avg",
    volume_upper_bound_factor=1.0,
):
    assert volume_upper_bound_factor >= 1.0
    assert conf_mode in ["volume", "prob", "fuse"]
    assert prob_mode in ["mean", "max", "center"]
    Z, Y, X = heatmap.shape

    # @TODO - carefully check the documents: https://github.com/seung-lab/connected-components-3d
    components = cc3d.connected_components(
        (heatmap >= conf_thres).cpu().numpy(), binary_image=True
    )
    stats = cc3d.statistics(components)
    num_components = len(stats["voxel_counts"])
    assert num_components >= 1
    if num_components == 1:
        # only background
        return torch.zeros((0, 4), dtype=torch.float32)

    voxel_counts = torch.tensor(stats["voxel_counts"], dtype=torch.float32)
    # the most common one should be background, on the first index
    assert voxel_counts.argmax() == 0

    # keep only large enough mass
    volumes = []
    component_idxs = []
    centroids = []
    for i in range(1, num_components):
        if voxel_counts[i] >= volume_thres:
            volumes.append(voxel_counts[i])
            component_idxs.append(i)
            centroids.append(stats["centroids"][i])
    volumes = torch.tensor(volumes, dtype=torch.float32)

    logger.debug(
        "\nSEGMENT DECODE: %d -> %d components, conf_mode=%s prob_mode=%s\n",
        num_components,
        len(volumes),
        conf_mode,
        prob_mode,
    )

    # calculate confident based on mass volume (voxel counts)
    volume_confs = volumes / (
        volume_thres * volume_upper_bound_factor
    )  # in range [1/volume_upper_bound_factor, inf]

    # calculate confidents based on probability map if needed
    if conf_mode in ["prob", "fuse"]:
        probs = []
        for component_idx, v, centroid in zip(component_idxs, volumes, centroids):
            if prob_mode == "center":
                z, y, x = centroid
                rz = min(max(0, round(z)), Z - 1)
                ry = min(max(0, round(y)), Y - 1)
                rx = min(max(0, round(x)), X - 1)
                prob = heatmap[rz, ry, rx]
            elif prob_mode == "mean":
                # prob = torch.sum(heatmap * (components_torch == component_idx)) / v
                prob = heatmap[components == component_idx].sum() / v
            elif prob_mode == "max":
                # prob = torch.max(heatmap * (components_torch == component_idx))
                prob = heatmap[components == component_idx].max()
            else:
                raise ValueError
            probs.append(prob.cpu())
        probs = torch.tensor(probs, dtype=torch.float32)
        assert len(probs) == len(volume_confs)

    if conf_mode == "volume":
        confs = volume_confs
    elif conf_mode == "prob":
        confs = probs
    elif conf_mode == "fuse":
        confs = (
            torch.minimum(volume_confs, torch.tensor(1.0, dtype=torch.float32)) * probs
        )
    else:
        raise ValueError

    if max_dets == 1:
        select_idx = confs.argmax()
        z, y, x = centroids[select_idx]
        conf = confs[select_idx]
        return torch.tensor([[z, y, x, conf]], dtype=torch.float32)
    else:
        raise NotImplementedError
