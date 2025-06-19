import logging
import os
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import torch
import time
from torch.nn import functional as F

logger = logging.getLogger(__name__)


WRONG_QUANTILE_TOMO_IDS = [
    "tomo_08bf73",
    "tomo_24a095",
    "tomo_37c426",
    "tomo_3a0914",
    "tomo_3b8291",
    "tomo_5b359d",
    "tomo_648adf",
    "tomo_67ff4e",
    "tomo_692081",
    "tomo_6b1fd3",
    "tomo_774aae",
    "tomo_9f918e",
    "tomo_ac4f0d",
    "tomo_b18127",
    "tomo_d8c917",
]


class MultithreadOpencvTomogramLoader:
    """
    Fast tomogram loader given directory of .jpg files.
    Multithread was enable to reduce latency and reduce IO blocking
    """

    def __init__(self, num_workers=8):
        self.num_workers = num_workers
        self.pool = ThreadPool(num_workers)

    def close(self):
        logger.info("[MultithreadOpencvTomogramLoader] Closing workers pool..")
        self.pool.close()

    def __del__(self):
        self.close()

    @staticmethod
    def load_img_thread_func(args):
        buffer, fpath = args
        cv2.imread(fpath, buffer, cv2.IMREAD_UNCHANGED)

    def load(self, tomo_dir, start=None, end=None, step=None):
        _slice = slice(start, end, step)
        fnames = sorted(os.listdir(tomo_dir))
        assert fnames == [f"slice_{j:04d}.jpg" for j in range(len(fnames))]
        fnames = fnames[_slice]
        fpaths = [os.path.join(tomo_dir, fname) for fname in fnames]
        img = cv2.imread(fpaths[0], cv2.IMREAD_UNCHANGED)
        assert len(img.shape) == 2  # grayscale
        tomo = np.empty((len(fpaths), *img.shape), dtype=np.uint8)
        tomo[0] = img
        self.pool.map(
            MultithreadOpencvTomogramLoader.load_img_thread_func,
            [(tomo[i], fpaths[i]) for i in range(1, len(fpaths))],
        )
        tomo_id = os.path.basename(tomo_dir).split(".")[0]
        if tomo_id in WRONG_QUANTILE_TOMO_IDS:
            tomo = tomo + 127
        return tomo


class OpencvTomogramLoader:
    """
    Tomogram loader given a directory of .jpg files sequentially
    """

    def __init__(self):
        pass

    def load(self, tomo_dir, start=None, end=None, step=None):
        _slice = slice(start, end, step)
        fnames = sorted(os.listdir(tomo_dir))
        assert fnames == [f"slice_{j:04d}.jpg" for j in range(len(fnames))]
        fnames = fnames[_slice]
        fpaths = [os.path.join(tomo_dir, fname) for fname in fnames]
        tomo = None
        for i, fpath in enumerate(fpaths):
            if i == 0:
                img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                assert len(img.shape) == 2  # grayscale
                tomo = np.empty(
                    (len(fpaths), *img.shape), dtype=img.dtype
                )  # ZYX or DHW
                tomo[0] = img
            else:
                cv2.imread(fpath, tomo[i], cv2.IMREAD_UNCHANGED)
        tomo_id = os.path.basename(tomo_dir).split(".")[0]
        if tomo_id in WRONG_QUANTILE_TOMO_IDS:
            tomo = tomo + 127
        return tomo

    def close(self):
        pass


def read_tomo(tomo_id, data_dir, io_backend="cv2", tomo_reader=None):
    """Return 3D uint8 tomogram tensor ZYX or DHW"""
    jpg_tomo_dir = (
        os.path.join(data_dir, "raw", "train", tomo_id)
        if tomo_id.startswith("tomo_")
        else os.path.join(data_dir, "external", "tomogram", tomo_id)
    )
    if io_backend == "cv2_seq":
        tomo = tomo_reader.load(tomo_dir=jpg_tomo_dir)
    elif io_backend == "cv2":
        tomo = tomo_reader.load(tomo_dir=jpg_tomo_dir)
    elif io_backend == "npy":
        tomo = np.load(os.path.join(data_dir, "processed", "npy", f"{tomo_id}.npy"))
    else:
        raise ValueError
    tomo = torch.from_numpy(tomo)
    logger.debug("LOAD TOMO: %s %s %s", tomo_id, tomo.shape, tomo.dtype)
    return tomo


@torch.inference_mode()
def spacing_torch(ori_tomo, target_shape, device="cuda", mode="trilinear"):
    """Spacing using torch.nn.interpolate"""
    if tuple(ori_tomo.shape) == tuple(target_shape):
        return ori_tomo
    # NOTE: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
    # shrink -> AREA, enlarge -> CUBIC or LINEAR
    assert mode in ["trilinear", "nearest", "area", "nearest-exact"]
    assert len(ori_tomo.shape) == 3
    ori_tomo = ori_tomo[None, None].to(device)  # 11ZYX
    if mode != "nearest":
        ori_tomo = ori_tomo.float()

    ##### ISSUES #####
    ### with interpolation mode `nearest`
    # RuntimeError: upsample_nearest3d only supports output tensors with less than INT_MAX elements, but got [1, 1, 1178, 1367, 1367]
    # ref: https://github.com/pytorch/pytorch/issues/144855
    ### with interpolation mode `trilinear` on GPU
    # RuntimeError: CUDA error: invalid configuration argument
    ##### CURRENT SIMPLE FIX: if CUDA-interpolation-kernel fails, use CPU instead
    try:
        start = time.time()
        spaced_tomo = F.interpolate(
            ori_tomo,
            size=target_shape,
            mode=mode,
            align_corners=False if mode == "trilinear" else None,
            recompute_scale_factor=False,
        )[0, 0]
        end = time.time()
        logger.debug("INTERPOLATE WITH MODE=%s TAKE %.2f sec", mode, end - start)
    except RuntimeError as e:
        logger.warning(
            "EXCEPTION with torch.interpolate():\n%s\nAttempt CPU interpolation..", e
        )
        spaced_tomo = F.interpolate(
            ori_tomo.cpu(),
            size=target_shape,
            scale_factor=None,
            mode=mode,
            align_corners=False if mode == "trilinear" else None,
            recompute_scale_factor=False,
        )[0, 0]
    if mode != "nearest":
        spaced_tomo = spaced_tomo.to(torch.uint8)  # ZYX
    spaced_tomo = spaced_tomo.cpu()
    assert spaced_tomo.dtype == torch.uint8
    return spaced_tomo


def load_spaced_val_sample(
    tomo_id,
    tomo_shape,
    data_dir,
    ori_kpts,
    ori_spacing,
    target_spacing,
    transform,
    io_backend="cv2",
    tomo_reader=None,
    interpolation_mode="trilinear",
):
    """
    Returns:
        Tuple of two element:
            tomo: 3D float32 tomogram, ZYX order
            kpts: (N, 10), ZYX order
    """
    tomo = read_tomo(tomo_id, data_dir, io_backend, tomo_reader)  # tensor, ZYX
    spacing_scale = np.array(
        [tgt / ori for ori, tgt in zip(ori_spacing, target_spacing)]
    )  # ZYX
    if transform is not None:
        # using MONAI transform
        data = {
            "image": tomo.permute(2, 1, 0)[None],  # ZYX -> 1XYZ = 1HWD
            "spacing_scale": spacing_scale[::-1],  # ZYX -> XYZ
            "kpts": ori_kpts[None],  # XYZ already
        }
        tdata = transform(data)
        tomo = tdata["image"][0].permute(2, 1, 0)  # 1XYZ -> ZYX
        assert tomo.dtype == torch.float32
        assert tuple(tomo.shape) == tomo_shape
        # pre-cast to uint8 to save RAM and faster memory transfer
        tomo = torch.clip(tomo, 0, 255, out=tomo).to(torch.uint8)
        kpts = tdata["kpts"]
        assert len(kpts.shape) == 3 and kpts.shape[0] == 1 and kpts.shape[2] == 10
        kpts = kpts[0, :, [2, 1, 0, 5, 4, 3, 8, 7, 6, 9]]  # XYZ order to ZYX order
    else:
        # Using torch's interpolate
        tomo = spacing_torch(tomo, tomo_shape, device="cpu", mode=interpolation_mode)
        # transform keypoint as well
        kpts = ori_kpts.clone()  # XYZ
        assert torch.all(kpts[:, 6:] == 0)
        kpts[:, :3] /= spacing_scale[::-1]  # mean XYZ
        kpts[:, 3:6] /= spacing_scale[::-1] ** 2  # variance XYZ
        assert len(kpts.shape) == 2 and kpts.shape[1] == 10
        kpts = kpts[:, [2, 1, 0, 5, 4, 3, 8, 7, 6, 9]]  # XYZ order to ZYX order

    return tomo, kpts


def load_spaced_val_sample_cache(
    tomo_id,
    tomo_shape,
    data_dir,
    ori_kpts,
    ori_spacing,
    target_spacing,
    interpolation_mode="trilinear",
    return_type="tensor",
):
    """
    Returns:
        Tuple of two element:
            tomo: 3D float32 tomogram, ZYX order
            kpts: (N, 10), ZYX order
    """
    tomo = np.load(
        os.path.join(
            data_dir,
            "processed",
            "spaced",
            f"{target_spacing}_{interpolation_mode}",
            f"{tomo_id}.npy",
        ),
        mmap_mode="r" if return_type == "mmap" else None,
    )  # tensor, ZYX
    assert (
        tomo.dtype == np.uint8 and tuple(tomo.shape) == tomo_shape
    ), f"{tomo.dtype} {tomo.shape} != {tomo_shape}"

    if return_type == "tensor":
        tomo = torch.from_numpy(tomo)
    spacing_scale = np.array(
        [tgt / ori for ori, tgt in zip(ori_spacing, target_spacing)]
    )  # ZYX
    # scale keypoints
    kpts = ori_kpts.clone()
    assert torch.all(kpts[:, 6:] == 0)
    kpts[:, :3] /= spacing_scale[::-1]  # mean
    kpts[:, 3:6] /= spacing_scale[::-1] ** 2  # variance
    assert len(kpts.shape) == 2 and kpts.shape[1] == 10
    kpts = kpts[:, [2, 1, 0, 5, 4, 3, 8, 7, 6, 9]]  # XYZ order to ZYX order
    return tomo, kpts
