import argparse
import gc
import glob
import json
import logging
import multiprocessing as mp
import os
import shutil
import time
import traceback
from typing import Tuple

import cv2
import numpy as np
import torch
import zarr
from cryoet_data_portal import Client, Run
from torch.nn import functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


SAVE_ATTRS = [
    "alignment_id",
    "ctf_corrected",
    "deposition_id",
    "fiducial_alignment_status",
    "https_mrc_file",
    "https_omezarr_dir",
    "id",
    "is_author_submitted",
    "is_portal_standard",
    "is_visualization_default",
    "key_photo_thumbnail_url",
    "key_photo_url",
    "name",
    "neuroglancer_config",
    "offset_x",
    "offset_y",
    "offset_z",
    "processing",
    "processing_software",
    "publications",
    "reconstruction_method",
    "reconstruction_software",
    "related_database_entries",
    "run_id",
    "s3_mrc_file",
    "s3_omezarr_dir",
    "scale_0_dimensions",
    "scale_1_dimensions",
    "scale_2_dimensions",
    "size_x",
    "size_y",
    "size_z",
    "tomogram_version",
    "tomogram_voxel_spacing_id",
    "voxel_spacing",
]

WRONG_QUANTILE_TOMO_IDS = [
    "mba2011-02-16-1",
    "mba2011-02-16-106",
    "mba2011-02-16-108",
    "mba2011-02-16-11",
    "mba2011-02-16-116",
    "mba2011-02-16-12",
    "mba2011-02-16-123",
    "mba2011-02-16-129",
    "mba2011-02-16-133",
    "mba2011-02-16-139",
    "mba2011-02-16-141",
    "mba2011-02-16-145",
    "mba2011-02-16-153",
    "mba2011-02-16-155",
    "mba2011-02-16-157",
    "mba2011-02-16-160",
    "mba2011-02-16-162",
    "mba2011-02-16-17",
    "mba2011-02-16-176",
    "mba2011-02-16-19",
    "mba2011-02-16-26",
    "mba2011-02-16-27",
    "mba2011-02-16-28",
    "mba2011-02-16-29",
    "mba2011-02-16-32",
    "mba2011-02-16-33",
    "mba2011-02-16-34",
    "mba2011-02-16-40",
    "mba2011-02-16-42",
    "mba2011-02-16-46",
    "mba2011-02-16-48",
    "mba2011-02-16-53",
    "mba2011-02-16-55",
    "mba2011-02-16-60",
    "mba2011-02-16-64",
    "mba2011-02-16-65",
    "mba2011-02-16-67",
    "mba2011-02-16-71",
    "mba2011-02-16-75",
    "mba2011-02-16-79",
    "mba2011-02-16-88",
    "mba2011-02-16-90",
    "mba2011-02-16-95",
]

TARGET_SPACING = 16.0


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/external/",
        help="Path to the saved external dataset",
    )
    parser.add_argument(
        "--tmp-download-dir",
        type=str,
        default="data/external/tmp/",
        help="Path to the temporary downloaded tomogram (before spacing)",
    )
    parser.add_argument(
        "--num-download-workers",
        type=int,
        default=8,
        help="Number of concurrent download workers",
    )
    parser.add_argument(
        "--num-process-workers",
        type=int,
        default=1,
        help="Number of concurrent process workers (spacing)",
    )
    return parser.parse_args()


args = parse_args()
OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)
TMP_DIR = args.tmp_download_dir
os.makedirs(TMP_DIR, exist_ok=True)
NUM_DOWNLOAD_WORKERS = args.num_download_workers
NUM_PROCESS_WORKERS = args.num_process_workers


@torch.inference_mode()
def spacing_torch(
    ori_tomo, ori_spacing, target_spacing, device="cuda", mode="trilinear"
):
    if tuple(ori_spacing) == tuple(target_spacing):
        return ori_tomo
    # NOTE: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
    # shrink -> AREA, enlarge -> CUBIC or LINEAR
    assert mode in ["trilinear", "nearest", "area", "nearest-exact"]
    assert len(ori_tomo.shape) == 3
    ori_tomo = ori_tomo[None, None].to(device)  # 11ZYX

    scale_factor = tuple(ori / tgt for ori, tgt in zip(ori_spacing, target_spacing))
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
            size=None,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=None,
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
            size=None,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=None,
            recompute_scale_factor=False,
        )[0, 0]
    spaced_tomo = spaced_tomo.cpu()
    return spaced_tomo


def process_tomogram(ori_tomo, ori_spacing, target_spacing, device="cpu"):
    # Norm
    lower, upper = np.percentile(ori_tomo, 0.87910068792905), np.percentile(
        ori_tomo, 99.06423106440042
    )
    ori_tomo = np.clip(ori_tomo, lower, upper)
    ori_tomo = (ori_tomo - lower) / (upper - lower)  # [0, 1] range
    ori_tomo = torch.from_numpy(ori_tomo)
    ret_tomo = spacing_torch(
        ori_tomo, ori_spacing, target_spacing, device=device, mode="trilinear"
    )
    del ori_tomo
    gc.collect()
    assert ret_tomo.dtype == torch.float32
    ret_tomo = torch.clip(255 * ret_tomo, 0, 255).to(torch.uint8).cpu().numpy()
    return ret_tomo


def process_worker(in_queue, worker_id):
    print(f"START NEW PROCESS WORKER NUMBER {worker_id}")
    while True:
        item = in_queue.get(block=True)
        if item is None:
            in_queue.put(None)
            print(f"[PROCESS WORKER {worker_id}] STOPPING..")
            return

        start = time.time()
        tomo_idx, tomo_tmp_dir, meta = item
        tomo_id = meta["tomo_id"]
        print(
            f"[PROCESS WORKER {worker_id}]Start process tomo number {tomo_idx} {tomo_id}"
        )
        # Load tomo
        zarr_tomo_path = glob.glob(os.path.join(tomo_tmp_dir, "*"))[0]
        tomo = zarr.open(zarr_tomo_path, mode="r")
        tomo = tomo[0][:]  # numpy

        ori_shape = tomo.shape
        meta["ori_shape"] = ori_shape
        meta["ori_dtype"] = str(tomo.dtype)

        # histogram analysis
        tomo_flat = tomo.flatten()
        tomo_min = float(tomo_flat.min())
        tomo_max = float(tomo_flat.max())
        hist, bin_edges = np.histogram(tomo_flat, bins=256, range=(tomo_min, tomo_max))
        meta["ori_histogram"] = hist.tolist()
        meta["ori_histogram_bins"] = bin_edges.tolist()
        meta["ori_tomo_min"] = tomo_min
        meta["ori_tomo_max"] = tomo_max
        meta["ori_tomo_mean"] = float(tomo.mean())
        meta["ori_tomo_std"] = float(tomo.std())

        # check if the original dtype if uint8 but saved as float32 -> wrong intensity value
        is_uint8_range = bool(
            (-128 <= tomo_min <= tomo_max <= 128)
            and np.abs(tomo - tomo.astype(int)).sum() == 0
        )
        meta["detected_as_uint8_range"] = is_uint8_range

        ############ FIX WRONG QUANTILE TOMOGRAM ##############
        # correct if needed, thanks https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/575028
        if is_uint8_range and tomo_id in WRONG_QUANTILE_TOMO_IDS:
            tomo = tomo.astype(np.uint8).astype(np.float32)
        #################################

        # Preprocess
        ori_spacing = meta["ori_spacing"]
        if TARGET_SPACING is None:
            target_spacing = ori_spacing
        else:
            target_spacing = max(ori_spacing, TARGET_SPACING)
        meta["target_spacing"] = target_spacing
        tomo = process_tomogram(tomo, [ori_spacing] * 3, [target_spacing] * 3)
        gc.collect()
        assert tomo.dtype == np.uint8
        meta["target_shape"] = tomo.shape
        # save as multiple .jpg image
        Z, Y, X = tomo.shape
        save_tomo_dir = os.path.join(OUT_DIR, "tomogram", tomo_id)
        try:
            shutil.rmtree(save_tomo_dir)
        except:
            pass
        os.makedirs(save_tomo_dir, exist_ok=False)
        for z in range(Z):
            img = tomo[z]
            img_save_path = os.path.join(save_tomo_dir, f"slice_{z:04d}.jpg")
            cv2.imwrite(img_save_path, img)

        meta_save_path = os.path.join(OUT_DIR, "meta", f"{tomo_id}.json")
        os.makedirs(os.path.dirname(meta_save_path), exist_ok=True)
        with open(meta_save_path, "w") as f:
            json.dump(meta, f)
        shutil.rmtree(tomo_tmp_dir)
        end = time.time()
        print(
            f"[PROCESS WORKER {worker_id}] Done process tomo number {tomo_idx} {tomo_id}, take {end - start:.2f} sec"
        )


def download_worker(in_queue, out_queue, worker_id):
    print("START NEW DOWNLOAD WORKER")
    while True:
        item = in_queue.get(block=True)
        if item[0] is None:
            num_remain_workers = item[1] - 1
            if num_remain_workers == 0:
                out_queue.put(None)
            in_queue.put((None, num_remain_workers))
            print(f"[DOWNLOAD WORKER {worker_id}] STOPPING..")
            return None
        tomo_idx, tomo_id = item
        print(
            f"[DOWNLOAD WORKER {worker_id}] Start download tomo number {tomo_idx} {tomo_id}"
        )
        start = time.time()
        client = Client()
        # ========= Process Run ==========
        run = Run.find(client, query_filters=[Run.name == tomo_id])
        if len(run) != 1:
            print("NUMBER OF RUNS ! 1:", tomo_id, len(run))
        if len(run) == 0:
            print("MISSING: ", tomo_id)
            continue
        else:
            run = run[0]

        # Download tomo
        try:
            if len(run.tomograms) != 1:
                print(tomo_id, len(run.tomograms))
            tomo = run.tomograms[0]
            meta = {
                "tomo_id": tomo_id,
            }
            ori_spacing = float(tomo.voxel_spacing)
            meta["ori_spacing"] = ori_spacing
            for attr in SAVE_ATTRS:
                meta[f"PORTAL__{attr}"] = getattr(tomo, attr, None)
            tomo_tmp_dir = os.path.join(TMP_DIR, tomo_id)
            try:
                shutil.rmtree(tomo_tmp_dir)
            except:
                pass
            os.makedirs(tomo_tmp_dir, exist_ok=False)
            tomo.download_omezarr(dest_path=tomo_tmp_dir)
            end = time.time()
            print(
                f"[DOWNLOAD WORKER {worker_id}] Done download tomo number {tomo_idx} {tomo_id} with spacing {ori_spacing}, take {end - start:.2f} sec"
            )
            out_queue.put((tomo_idx, tomo_tmp_dir, meta))
        except Exception as e:
            print("EXCEPTION OCCUR:", e)
            print(traceback.format_exc())


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/external/brendanartley_external_labels.csv")
    print(df)
    TOMO_IDS = sorted(list(df["tomo_id"].unique()))
    # this tomogram contains 0 run ???
    # -> 1288 - 1 = 1287 tomograms in total
    TOMO_IDS.remove("mba2011-07-18-1")
    print("TOTAL NUMBER OF TOMOS:", len(TOMO_IDS))

    tomo_id_queue = mp.Queue()
    tomo_queue = mp.Queue(32)

    for tomo_idx, tomo_id in enumerate(TOMO_IDS):
        tomo_id_queue.put((tomo_idx, tomo_id))
    tomo_id_queue.put((None, NUM_DOWNLOAD_WORKERS))
    time.sleep(1.0)

    all_workers = []
    for worker_id in range(NUM_DOWNLOAD_WORKERS):
        worker = mp.Process(
            group=None,
            target=download_worker,
            name=f"download_worker_{worker_id}",
            args=(tomo_id_queue, tomo_queue, worker_id),
        )
        worker.start()
        all_workers.append(worker)

    for worker_id in range(NUM_PROCESS_WORKERS):
        worker = mp.Process(
            None,
            target=process_worker,
            name=f"process_worker_{worker_id}",
            args=(tomo_queue, worker_id),
        )
        worker.start()
        all_workers.append(worker)

    print("WAITING ALL WORKERS TO BE FINISHED..")
    for worker in all_workers:
        worker.join()
