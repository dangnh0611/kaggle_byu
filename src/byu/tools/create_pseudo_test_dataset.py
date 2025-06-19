"""
Create pseudo test dataset with larger resolution for testing to estimate runtime and prevent unexpected errors such as OOM
"""

import argparse
import multiprocessing as mp
import os
import queue
import random
from copy import deepcopy

import cv2
import numpy as np
import polars as pl
import torch
from torch.nn import functional as F
from tqdm import tqdm

from byu.data.io import MultithreadOpencvTomogramLoader

df = pl.scan_csv("/home/dangnh36/datasets/.comp/byu/processed/gt_v2.csv").collect()
tomo_loader = MultithreadOpencvTomogramLoader(8)
device = torch.device("cpu")

os.makedirs(
    "/home/dangnh36/datasets/.comp/byu/processed/pseudo_test/tomograms", exist_ok=False
)

new_rows = []
for row_idx, row in enumerate(df.sample(20).iter_rows(named=True)):
    print(row)
    tomo_id = row["tomo_id"]

    tomo_dir = os.path.join("/home/dangnh36/datasets/.comp/byu/raw/train/", tomo_id)

    # read ori tomo
    tomo = tomo_loader.load(tomo_dir)
    ori_shape = tomo.shape  # ZYX
    print(ori_shape)
    MAX_SHAPE = (450, 1500, 1200)
    max_new_shape = [
        (random.randrange(int(new * 1.2), int(new * 1.5))) for new in MAX_SHAPE
    ]
    print("max new shape:", max_new_shape)
    scale = max([new / old for new, old in zip(max_new_shape, ori_shape)])
    print("scale:", scale)

    new_tomo = F.interpolate(
        torch.from_numpy(tomo).to(device)[None, None].float(),
        size=None,
        scale_factor=scale,
        mode="trilinear",
        align_corners=None,
        recompute_scale_factor=False,
    )[0, 0]
    new_tomo = torch.clip(new_tomo, 0, 255.0).to(torch.uint8).cpu().numpy()
    print("Return", new_tomo.shape, new_tomo.dtype)

    # scale the annotations too
    new_row = deepcopy(row)
    new_row["Z"] = new_tomo.shape[0]
    new_row["Y"] = new_tomo.shape[1]
    new_row["X"] = new_tomo.shape[2]
    new_row["voxel_spacing"] /= scale
    new_row["V"] *= scale**3
    if row["motor_z"] != -1:
        new_row["motor_z"] *= scale
        new_row["motor_y"] *= scale
        new_row["motor_x"] *= scale
    new_rows.append(new_row)
    print("-----------------------------------\n\n")

    tomo_dir = (
        f"/home/dangnh36/datasets/.comp/byu/processed/pseudo_test/tomograms/{tomo_id}"
    )
    os.makedirs(tomo_dir, exist_ok=False)
    for z_idx in tqdm(range(new_tomo.shape[0]), desc=f"Saving {tomo_id}"):
        slice_img = new_tomo[z_idx]
        fname = os.path.join(tomo_dir, f"slice_{z_idx:04d}.jpg")
        cv2.imwrite(fname, slice_img)

new_df = pl.DataFrame(new_rows)
new_df = new_df.select(
    pl.col(
        [
            "tomo_id",
            "Z",
            "Y",
            "X",
            "voxel_spacing",
            "ori_num_motors",
            "num_motors",
            "motor_z",
            "motor_y",
            "motor_x",
        ]
    )
)
new_df.write_csv("/home/dangnh36/datasets/.comp/byu/processed/pseudo_test/gt.csv")
