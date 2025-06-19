import warnings

warnings.simplefilter("ignore")
import gc
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import sys
import threading
import time

import cc3d
import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate
from tqdm import tqdm
from yagm.tasks.base_task import BaseTask
from yagm.transforms.keypoints.decode import decode_heatmap_3d, decode_segment_mask_3d
from yagm.transforms.sliding_window import get_sliding_patch_positions
from yagm.transforms.tta_3d import build_tta
from yagm.utils import hydra as hydra_utils
from yagm.utils import lightning as l_utils
from yagm.utils.concurrent import ShmTensor

# Setting up custom logging
from yagm.utils.logging import init_logging, setup_logging

from byu.data.io import MultithreadOpencvTomogramLoader
from byu.inference import zoo
from byu.utils.data import SubmissionDataFrame
from byu.utils.metrics import compute_metrics, kaggle_score
from byu.utils.viz import viz_byu

try:
    hydra_utils.init_hydra()
except:
    print("SKIP RE-INIT HYDRA")

### SETUP LOGGER FOR IPYTHON ###
# ref: https://github.com/ipython/ipykernel/issues/111
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)
# Create formatter and add it to the handler
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
# Set STDERR handler as the only handler
logger.handlers = [handler]


##################################################
###### MODEL SIGNATURES OVERRIDES
##################################################

EXP1_X3DM = {
    "model": "EXP1_X3DM",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        # {
        #     "fold": 0,
        #     "config_path": "EXP1_X3DM_config.yaml",
        #     "ckpt_path": "EXP1_X3DM_ep4_step30000.ckpt",
        #     "ema": 0.99
        # },
        # {
        #     "fold": 0,
        #     "config_path": "EXP1_X3DM_config.yaml",
        #     "ckpt_path": "EXP1_X3DM_ep3_step27000.ckpt",
        #     "ema": 0.99
        # },
        {
            "fold": 0,
            "config_path": "EXP1_X3DM_config.yaml",
            "ckpt_path": "EXP1_X3DM_ep3_step24000.ckpt",
            "ema": 0.99,
        },
    ],
}

EXP2_X3DL = {
    "model": "EXP2_X3DL",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "EXP2_X3DL_config.yaml",
            "ckpt_path": "EXP2_X3DL_ep4_step30000.ckpt",
            "ema": 0.99,
        },
        # {
        #     "fold": 0,
        #     "config_path": "EXP2_X3DL_config.yaml",
        #     "ckpt_path": "EXP2_X3DL_ep3_step26000.ckpt",
        #     "ema": 0.99
        # },
        # {
        #     "fold": 0,
        #     "config_path": "EXP2_X3DL_config.yaml",
        #     "ckpt_path": "EXP2_X3DL_ep3_step22000.ckpt",
        #     "ema": 0.99
        # },
    ],
}

EXP3_X3DM_COOLDOWNFINETUNE = {
    "model": "X3DM_COOLDOWNFINETUNE",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "EXP3_X3DL_COOLDOWNFINETUNE_config.yaml",
            "ckpt_path": "EXP3_X3DL_COOLDOWNFINETUNE_ep1_step5000_val_Fbeta0.934035_val_PAP0.942526.ckpt",
            "ema": 0.99,
        },
    ],
}

EXP4_R101 = {
    "model": "EXP4_R101",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        # {
        #     "fold": 0,
        #     "config_path": "EXP4_R101_config.yaml",
        #     "ckpt_path": "EXP4_R101_ep4_step30000.ckpt",
        #     "ema": 0.99
        # },
        {
            "fold": 0,
            "config_path": "EXP4_R101_config.yaml",
            "ckpt_path": "EXP4_R101_ep3_step24000.ckpt",
            "ema": 0.99,
        },
    ],
}

EXP5_R50 = {
    "model": "EXP5_R50",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        # {
        #     "fold": 0,
        #     "config_path": "EXP5_R50_config.yaml",
        #     "ckpt_path": "EXP5_R50_ep4_step30000.ckpt",
        #     "ema": 0.99
        # },
        {
            "fold": 0,
            "config_path": "EXP5_R50_config.yaml",
            "ckpt_path": "EXP5_R50_ep3_step26000.ckpt",
            "ema": 0.99,
        },
    ],
}

EXP6_X3DM_MIXUPMIX1 = {
    "model": "EXP6_X3DM_MIXUPMIX1",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "EXP6_X3DM_MIXUPMIX1_config.yaml",
            "ckpt_path": "EXP6_X3DM_MIXUPMIX1_ep4_step30000.ckpt",
            "ema": 0.99,
        },
    ],
}

EXP7_DENSENET121 = {
    "model": "EXP7_DENSENET121",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "EXP7_DENSENET121_config.yaml",
            "ckpt_path": "EXP7_DENSENET121_ep4_step30000.ckpt",
            "ema": 0.99,
        },
    ],
}

EXP8_CONVNEXT_AVG = {
    "model": "EXP8_CONVNEXT_AVG",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "EXP8_CONVNEXT_AVG_config.yaml",
            "ckpt_path": "EXP8_CONVNEXT_AVG_ep4_step30000.ckpt",
            "ema": 0.99,
        },
        # {
        #     "fold": 0,
        #     "config_path": "EXP8_CONVNEXT_AVG_config.yaml",
        #     "ckpt_path": "EXP8_CONVNEXT_AVG_ep3_step26000.ckpt",
        #     "ema": 0.99
        # },
        # {
        #     "fold": 0,
        #     "config_path": "EXP8_CONVNEXT_AVG_config.yaml",
        #     "ckpt_path": "EXP8_CONVNEXT_AVG_ep3_step22000.ckpt",
        #     "ema": 0.99
        # },
    ],
}

EXP9_CONVNEXT_AVG_LSTM = {
    "model": "EXP9_CONVNEXT_AVG_LSTM",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "EXP9_CONVNEXT_AVG_LSTM_config.yaml",
            "ckpt_path": "EXP9_CONVNEXT_AVG_LSTM_ep4_step30000.ckpt",
            "ema": 0.99,
        },
        #  {
        #     "fold": 0,
        #     "config_path": "EXP9_CONVNEXT_AVG_LSTM_config.yaml",
        #     "ckpt_path": "EXP9_CONVNEXT_AVG_LSTM_ep3_step26000.ckpt",
        #     "ema": 0.99
        # },
        #  {
        #     "fold": 0,
        #     "config_path": "EXP9_CONVNEXT_AVG_LSTM_config.yaml",
        #     "ckpt_path": "EXP9_CONVNEXT_AVG_LSTM_ep3_step22000.ckpt",
        #     "ema": 0.99
        # },
    ],
}


EXP10_R101_LR1e4 = {
    "model": "EXP10_R101_LR1e4",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        # {
        #     "fold": 0,
        #     "config_path": "EXP10_R101_LR1e4_config.yaml",
        #     "ckpt_path": "/kaggle/input/byu-final-dataset/EXP10_R101_LR1e4_ep4_step30000.ckpt",
        #     "ema": 0.99
        # },
        # {
        #     "fold": 0,
        #     "config_path": "EXP10_R101_LR1e4_config.yaml",
        #     "ckpt_path": "/kaggle/input/byu-final-dataset/EXP10_R101_LR1e4_ep3_step26000.ckpt",
        #     "ema": 0.99
        # },
        # {
        #     "fold": 0,
        #     "config_path": "EXP10_R101_LR1e4_config.yaml",
        #     "ckpt_path": "/kaggle/input/byu-final-dataset/EXP10_R101_LR1e4_ep3_step22000.ckpt",
        #     "ema": 0.99
        # },
        {
            "fold": 0,
            "config_path": "EXP10_R101_LR1e4_config.yaml",
            "ckpt_path": "EXP10_R101_LR1e4_ep2_step16000.ckpt",
            "ema": 0.99,
        },
    ],
}

EXP11_MAXVIT_PICO_AVG_LSTM = {
    "model": "EXP11_MAXVIT_PICO_AVG_LSTM",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        # {
        #     "fold": 0,
        #     "config_path": "EXP11_MAXVIT_PICO_AVG_LSTM_config.yaml",
        #     "ckpt_path": "EXP11_MAXVIT_PICO_AVG_LSTM_ep4_step30000.ckpt",
        #     "ema": 0.99
        # },
        # {
        #     "fold": 0,
        #     "config_path": "EXP11_MAXVIT_PICO_AVG_LSTM_config.yaml",
        #     "ckpt_path": "/EXP11_MAXVIT_PICO_AVG_LSTM_ep3_step24000.ckpt",
        #     "ema": 0.99
        # },
        {
            "fold": 0,
            "config_path": "EXP11_MAXVIT_PICO_AVG_LSTM_config.yaml",
            "ckpt_path": "EXP11_MAXVIT_PICO_AVG_LSTM_ep2_step18000.ckpt",
            "ema": 0.99,
        },
    ],
}


EXP12_EFFB0_AVG_LSTM = {
    "model": "EXP12_EFFB0_AVG_LSTM",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "EXP12_EFFB0_AVG_LSTM_config.yaml",
            "ckpt_path": "EXP12_EFFB0_AVG_LSTM_ep4_step30000.ckpt",
            "ema": 0.99,
        },
        # {
        #     "fold": 0,
        #     "config_path": "EXP12_EFFB0_AVG_LSTM_config.yaml",
        #     "ckpt_path": "/EXP12_EFFB0_AVG_LSTM_ep3_step24000.ckpt",
        #     "ema": 0.99
        # },
        # {
        #     "fold": 0,
        #     "config_path": "EXP12_EFFB0_AVG_LSTM_config.yaml",
        #     "ckpt_path": "EXP12_EFFB0_AVG_LSTM_ep2_step18000.ckpt",
        #     "ema": 0.99
        # },
    ],
}

EXP13_RESNEXT50_32x4d = {
    "model": "EXP13_RESNEXT50_32x4d",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "EXP13_RESNEXT50_32x4d_config.yaml",
            "ckpt_path": "EXP13_RESNEXT50_32x4d_ep4_step30000.ckpt",
            "ema": 0.99,
        },
        # {
        #     "fold": 0,
        #     "config_path": "EXP13_RESNEXT50_32x4d_config.yaml",
        #     "ckpt_path": "EXP13_RESNEXT50_32x4d_ep3_step24000.ckpt",
        #     "ema": 0.99
        # },
        # {
        #     "fold": 0,
        #     "config_path": "EXP13_RESNEXT50_32x4d_config.yaml",
        #     "ckpt_path": "EXP13_RESNEXT50_32x4d_ep2_step18000.ckpt",
        #     "ema": 0.99
        # },
    ],
}

EXP14_3DEFFB0 = {
    "model": "EXP14_3DEFFB0",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "EXP14_3DEFFB0_config.yaml",
            "ckpt_path": "EXP14_3DEFFB0_ep4_step30000.ckpt",
            "ema": 0.99,
        },
        # {
        #     "fold": 0,
        #     "config_path": "EXP14_3DEFFB0_config.yaml",
        #     "ckpt_path": "EXP14_3DEFFB0_ep3_step26000.ckpt",
        #     "ema": 0.99
        # },
        # {
        #     "fold": 0,
        #     "config_path": "EXP14_3DEFFB0_config.yaml",
        #     "ckpt_path": "EXP14_3DEFFB0_ep3_step22000.ckpt",
        #     "ema": 0.99
        # },
    ],
}

EXP16_X3DM_ALL_TRAIN_EXT = {
    "model": "EXP16_X3DM_ALL_TRAIN_EXT",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "EXP16_X3DM_ALL_TRAIN_EXT_config.yaml",
            "ckpt_path": "EXP16_X3DM_ALL_TRAIN_EXT_ep1_step30000.ckpt",
            "ema": 0.99,
        },
        {
            "fold": 0,
            "config_path": "EXP16_X3DM_ALL_TRAIN_EXT_config.yaml",
            "ckpt_path": "EXP16_X3DM_ALL_TRAIN_EXT_ep1_step26000.ckpt",
            "ema": 0.99,
        },
        {
            "fold": 0,
            "config_path": "EXP16_X3DM_ALL_TRAIN_EXT_config.yaml",
            "ckpt_path": "EXP16_X3DM_ALL_TRAIN_EXT_ep1_step22000.ckpt",
            "ema": 0.99,
        },
        # {
        #     "fold": 0,
        #     "config_path": "EXP16_X3DM_ALL_TRAIN_EXT_config.yaml",
        #     "ckpt_path": "EXP16_X3DM_ALL_TRAIN_EXT_ep0_step18000.ckpt",
        #     "ema": 0.99
        # },
        # {
        #     "fold": 0,
        #     "config_path": "EXP16_X3DM_ALL_TRAIN_EXT_config.yaml",
        #     "ckpt_path": "/kaggle/working/swa.pth",
        #     "ema": 0.99
        # },
    ],
}

EXP18_X3DM_PAN_ALLGTV2 = {
    "model": "EXP18_X3DM_PAN_ALLGTV2",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "TORCH_COMBINED_WEIGHT_PATH.pth",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "EXP18_X3DM_PAN_ALLGTV2_config.yaml",
            "ckpt_path": "EXP18_X3DM_PAN_ALLGTV2_ep1_step30000.ckpt",
            "ema": 0.99,
        },
        # {
        #     "fold": 0,
        #     "config_path": "EXP18_X3DM_PAN_ALLGTV2_config.yaml",
        #     "ckpt_path": "EXP18_X3DM_PAN_ALLGTV2_ep1_step26000.ckpt",
        #     "ema": 0.99
        # },
        # {
        #     "fold": 0,
        #     "config_path": "EXP18_X3DM_PAN_ALLGTV2_config.yaml",
        #     "ckpt_path": "EXP18_X3DM_PAN_ALLGTV2_ep1_step22000.ckpt",
        #     "ema": 0.99
        # },
    ],
}


JOBS = [
    {
        "sigs": [EXP18_X3DM_PAN_ALLGTV2],
        "names": ["EXP18_X3DM_PAN_ALLGTV2"],
        "weights": [1.0],
    },
]

##################################################
###### GLOBAL VARS
##################################################
LOAD_COMBINED_WEIGHT = True
_CV2_TOMO_LOADER_NUM_THREADS = 8


IO_BACKEND = "cv2"
BASE_SPACING = 13.1  # avg 15.6
# TARGET_SPACINGS = [13.1, 16.0, 19.7]
TARGET_SPACINGS = [[16.0, 16.0, 16.0]]
TOMO_SPACING_DEVICE = "cpu"
TOMO_SPACING_MODE = "trilinear"

HEATMAP_ENSEMBLE = True
HEATMAP_AGG_MODE = "avg"
HEATMAP_AGG_LOGITS = True
HEATMAP_INTERPOLATION_MODE = "trilinear"
FORWARD_SAVE_MEMORY = False
BATCH_SIZE = 1
HEATMAP_STRIDE = 8

# MASK CC3D DECODING
DECODE_CC3D_CONF_THRES = 0.05
DECODE_CC3D_RADIUS_FACTOR = 0.1
DECODE_CC3D_CONF_MODE = "prob"  # volume | prob | fuse
DECODE_CC3D_PROB_MODE = "center"  # center | mean | max
# HEATMAP NMS DECODING
DECODE_NMS_BLUR_SIGMA = None
###### MAIN DECODE PARAMS ######
DECODE_METHOD = "nms"  # nms, cc3d
DECODE_CONF_THRES = 0.01
QUANTILE_THRES = 0.55

VIZ_ENABLE = False


# automatic determine the MODE
if os.path.isdir("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"):
    MODE = "KAGGLE"
else:
    MODE = "LOCAL"
# manual override
# MODE = "KAGGLE_VAL"

# LOCAL DEVELOPMENT ENV
if MODE == "LOCAL":
    DEVICES = [0, 0]
    ASSETS_DIR = "assets/"
    WORKING_DIR = "outputs/submit/working/"

    DATA_DIR = "/home/dangnh36/datasets/.comp/byu/raw/train"
    VAL_GT_CSV_PATH = "/home/dangnh36/datasets/.comp/byu/processed/gt_v2.csv"
    with open(
        "/home/dangnh36/datasets/.comp/byu/processed/cv/v2/skf4_rd42.json", "r"
    ) as f:
        cv_meta = json.load(f)
        ALL_TOMO_IDS = cv_meta["folds"][1]["val"][:29]  # fold 0

    # DATA_DIR = "/home/dangnh36/datasets/.comp/byu/processed/pseudo_test/tomograms"
    # VAL_GT_CSV_PATH = "/home/dangnh36/datasets/.comp/byu/processed/pseudo_test/gt.csv"
    # ALL_TOMO_IDS = sorted(os.listdir(DATA_DIR)) * 10

    GT_DF = pd.read_csv(VAL_GT_CSV_PATH)
    GT_DF = GT_DF[GT_DF["tomo_id"].isin(ALL_TOMO_IDS)].reset_index(drop=True)
    _tomo2spacing = {}
    for i, row in GT_DF.iterrows():
        _tomo2spacing[row["tomo_id"]] = row["voxel_spacing"]
    ALL_TOMO_SPACINGS = [_tomo2spacing[tomo_id] for tomo_id in ALL_TOMO_IDS]

    # TOMO_PRODUCER_NUM_WORKERS = 8
    # TOMO_PRODUCER_PREFETCH = 8
    # PATCHES_PRODUCER_NUM_WORKERS = 2
    # PATCHES_PRODUCER_PREFETCH = 8
    # DATALOADER_PREFETCH = 8

    TOMO_PRODUCER_NUM_WORKERS = 1
    TOMO_PRODUCER_PREFETCH = 1
    PATCHES_PRODUCER_NUM_WORKERS = 1
    PATCHES_PRODUCER_PREFETCH = 2
    DATALOADER_PREFETCH = 8

# VALIDATION ON KAGGLE
elif MODE == "KAGGLE_VAL":
    DEVICES = [0, 1]
    ASSETS_DIR = "/kaggle/input/byu-final-dataset/"
    DATA_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train"
    WORKING_DIR = "/kaggle/working/"
    VAL_GT_CSV_PATH = "/kaggle/input/byu-checkpoints/gt_v2.csv"
    with open("/kaggle/input/byu-checkpoints/skf4_rd42.json", "r") as f:
        cv_meta = json.load(f)
        ALL_TOMO_IDS = cv_meta["folds"][0]["val"][:50]  # fold 0, top 10 first tomo
    GT_DF = pd.read_csv(VAL_GT_CSV_PATH)
    GT_DF = GT_DF[GT_DF["tomo_id"].isin(ALL_TOMO_IDS)].reset_index(drop=True)
    _tomo2spacing = {}
    for i, row in GT_DF.iterrows():
        _tomo2spacing[row["tomo_id"]] = row["voxel_spacing"]
    ALL_TOMO_SPACINGS = [_tomo2spacing[tomo_id] for tomo_id in ALL_TOMO_IDS]

    TOMO_PRODUCER_NUM_WORKERS = 1
    TOMO_PRODUCER_PREFETCH = 1
    PATCHES_PRODUCER_NUM_WORKERS = 1
    PATCHES_PRODUCER_PREFETCH = 2
    DATALOADER_PREFETCH = 8

# PRIVATE TEST SUBMISSION
elif MODE == "KAGGLE":
    DEVICES = [0, 1]
    ASSETS_DIR = "/kaggle/input/byu-final-dataset/"
    DATA_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test"
    WORKING_DIR = "/kaggle/working/"
    ALL_TOMO_IDS = sorted(os.listdir(DATA_DIR))
    ALL_TOMO_SPACINGS = [BASE_SPACING] * len(ALL_TOMO_IDS)

    TOMO_PRODUCER_NUM_WORKERS = 1
    TOMO_PRODUCER_PREFETCH = 1
    PATCHES_PRODUCER_NUM_WORKERS = 1
    PATCHES_PRODUCER_PREFETCH = 2
    DATALOADER_PREFETCH = 8

else:
    raise ValueError

TMP_CSV_DIR = os.path.join(WORKING_DIR, "tmp_csv")
TMP_VIZ_DIR = os.path.join(WORKING_DIR, "tmp_viz")

# RECHECK SOME CONDITIONS
assert len(ALL_TOMO_IDS) == len(ALL_TOMO_SPACINGS)


###################################################
###### FUNCTIONS/CLASSES DEFINITION
###################################################
def exception_printer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception("Exception %s in %s: %s", type(e), func.__name__, e)
            import traceback

            traceback.print_exc()
            raise e

    return wrapper


def clear_dir_content(folder_path):
    if os.path.isdir(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)


# @TODO - Multi-folds + Multi-TTAs Wrapper in once to maximize performance (reduce CUDA synchronization as much as posible)
# but this current approach is simple and clean, especially when applying TensorRT to convert one engine for each multi-folds model
class KFoldModelWrapper(nn.Module):
    def __init__(
        self,
        sig,
        base_dir,
        load_combined_weight=False,
        heatmap_agg_mode="avg",
        save_memory=False,
        act="sigmoid",
    ):
        super().__init__()
        assert heatmap_agg_mode in ["avg", "max"]
        self.heatmap_agg_mode = heatmap_agg_mode
        self.save_memory = save_memory
        fold_metas = sig["fold_metas"]

        if load_combined_weight:
            combined_weight_path = os.path.join(
                base_dir, sig["torch_combined_weight_path"]
            )
            if not os.path.isfile(combined_weight_path):
                logger.info(
                    "Load combined weight from %s failed (file not exist). Attempt load weight for each fold individually..",
                    combined_weight_path,
                )
                load_combined_weight = False

        fold_models = []
        for fold_idx, meta in enumerate(fold_metas):
            fold_idx = meta["fold"]
            cfg = OmegaConf.load(os.path.join(base_dir, meta["config_path"]))
            # assert cfg.cv.fold_idx == fold_idx
            cfg.ema.val_decays = [meta["ema"] if meta["ema"] is not None else 0]
            cfg.model.head.ms = [0]
            cfg.ckpt.strict = True
            # @TODO - remove this
            if (
                "x3d" in cfg.model.encoder._target_
                or "i3d" in cfg.model.encoder._target_
            ):
                cfg.model.encoder.pretrained = None
            elif "smp" in cfg.model.encoder._target_:
                cfg.model.encoder.weights = None
                if "resnet101" in cfg.model.encoder.model_name:
                    cfg.model.encoder.model_name = "resnet101"
            else:
                pass

            # for 2d encoder
            try:
                cfg.model.encoder.encoder_2d.pretrained = False
            except:
                pass
            try:
                if (
                    cfg.model.encoder.encoder_2d.model_name
                    == "convnext_nano.r384_ad_in12k"
                ):
                    cfg.model.encoder.encoder_2d.model_name = "convnext_nano"
            except:
                pass

            task: BaseTask = l_utils.build_task(cfg)
            if not load_combined_weight:
                ckpt_path = (
                    meta["ckpt_path"]
                    if os.path.isfile(meta["ckpt_path"])
                    else os.path.join(base_dir, meta["ckpt_path"])
                )
                l_utils.load_lightning_state_dict(
                    model=task,
                    ckpt_path=ckpt_path,
                    cfg=cfg,
                )
                logger.info(f"Loaded state dict for fold {fold_idx}")
            model = task.model
            fold_models.append(model)
        self.fold_models = nn.ModuleList(fold_models)
        self.num_fold_models = len(self.fold_models)

        if load_combined_weight:
            self.load_state_dict(torch.load(combined_weight_path))
            logger.info(f"Loaded combined weight from %s", combined_weight_path)

        self.act_name = act
        if act == "sigmoid":
            self.act = F.sigmoid
        elif act == "identity":
            self.act = lambda x: x
        else:
            raise ValueError

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.fold_models[0].__class__.__name__}, heatmap_agg_mode={self.heatmap_agg_mode}, save_memory={self.save_memory}, act={self.act_name})"

    def _forward_save_memory(self, x):
        agg_heatmap = 0.0
        for model_idx, model in enumerate(self.fold_models):
            out_heatmap = torch.nan_to_num(
                self.act(model(x)[0]), nan=0.0, posinf=1.0, neginf=0.0
            )  # (B, 1, D, H, W)
            # inplace ops, so force CUDA sync
            if self.heatmap_agg_mode == "avg":
                agg_heatmap = agg_heatmap + out_heatmap
            elif self.heatmap_agg_mode == "max":
                agg_heatmap = torch.maximum(
                    agg_heatmap,
                    out_heatmap,
                    out=None if model_idx == 0 else agg_heatmap,
                )
            else:
                raise ValueError
        if self.heatmap_agg_mode == "avg" and self.num_fold_models > 1:
            agg_heatmap /= self.num_fold_models
        return agg_heatmap

    def _forward_fast(self, x):
        # async, just store reference in output list
        out_heatmaps = [
            torch.nan_to_num(self.act(model(x)[0]), nan=0.0, posinf=1.0, neginf=0.0)
            for model in self.fold_models
        ]

        ### Method 1: this increase memory consumtion, caused by torch.stack()
        # if self.heatmap_agg_mode == 'avg':
        #     agg_heatmap = torch.stack(out_heatmaps).mean(dim=0)
        # elif self.heatmap_agg_mode == 'max':
        #     agg_heatmap = torch.stack(out_heatmaps).max(dim=0)
        # else:
        #     raise ValueError

        ### Method 2: just aggregate one by one
        agg_heatmap = out_heatmaps[0]
        for i in range(1, self.num_fold_models):
            if self.heatmap_agg_mode == "avg":
                agg_heatmap += out_heatmaps[i]
            elif self.heatmap_agg_mode == "max":
                agg_heatmap = torch.maximum(
                    agg_heatmap, out_heatmaps[i], out=agg_heatmap
                )
            else:
                raise ValueError
        if self.heatmap_agg_mode == "avg" and self.num_fold_models > 1:
            agg_heatmap /= self.num_fold_models
        return agg_heatmap

    def forward(self, x):
        if self.save_memory:
            return self._forward_save_memory(x)
        else:
            return self._forward_fast(x)


class TTAModelWrapper(nn.Module):
    def __init__(self, model, tta_names, heatmap_agg_mode="avg", save_memory=False):
        super().__init__()
        assert heatmap_agg_mode in ["avg", "max"]
        self.model = model
        self.heatmap_agg_mode = heatmap_agg_mode
        self.save_memory = save_memory

        assert len(tta_names) == len(set(tta_names))
        self.tta_names = tta_names
        self.tta_transforms = [
            build_tta(
                tta_name,
                weight=1.0,
                ori_dims="zyx",
            )
            for tta_name in tta_names
        ]
        self.num_ttas = len(tta_names)

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model}, ttas={self.tta_names}, heatmap_agg_mode={self.heatmap_agg_mode}, save_memory={self.save_memory})"

    def _forward_save_memory(self, x):
        """Save memory of caching intermediate TTA-ed heatmap, only keep one
        aggregated heatmap at a time, but force some CUDA synchronization
        on inplace operations, so may affect overall performance.
        """
        agg_heatmap = 0
        for tta_idx, tta_tf in enumerate(self.tta_transforms):
            tta_x, _w = tta_tf.transform(x)
            tta_heatmap = self.model(tta_x)
            out_heatmap, _w = tta_tf.invert(tta_heatmap)
            if self.heatmap_agg_mode == "avg":
                agg_heatmap += out_heatmap
            elif self.heatmap_agg_mode == "max":
                agg_heatmap = torch.maximum(
                    agg_heatmap, out_heatmap, out=None if tta_idx == 0 else agg_heatmap
                )
            else:
                raise ValueError
        if self.heatmap_agg_mode == "avg" and self.num_ttas > 1:
            agg_heatmap = agg_heatmap / self.num_ttas
        return agg_heatmap

    def _forward_fast(self, x):
        # async, just store reference in output list
        out_heatmaps = [
            tta_tf.invert(self.model(tta_tf.transform(x)[0]))[0]
            for tta_tf in self.tta_transforms
        ]

        # ## Method 1: this increase memory consumtion, caused by torch.stack()
        # if self.heatmap_agg_mode == 'avg':
        #     agg_heatmap = torch.stack(out_heatmaps).mean(dim=0)
        # elif self.heatmap_agg_mode == 'max':
        #     agg_heatmap = torch.stack(out_heatmaps).max(dim=0)
        # else:
        #     raise ValueError

        ### Method 2: just aggregate one by one
        agg_heatmap = out_heatmaps[0]
        for i in range(1, self.num_ttas):
            if self.heatmap_agg_mode == "avg":
                agg_heatmap += out_heatmaps[i]
            elif self.heatmap_agg_mode == "max":
                agg_heatmap = torch.maximum(
                    agg_heatmap, out_heatmaps[i], out=agg_heatmap
                )
            else:
                raise ValueError
        if self.heatmap_agg_mode == "avg" and self.num_ttas > 1:
            agg_heatmap /= self.num_ttas
        return agg_heatmap

    def forward(self, x):
        # simple avg heatmap of multiple TTAs, no weights
        if self.save_memory:
            return self._forward_save_memory(x)
        else:
            return self._forward_fast(x)


def decode_heatmap(pred_heatmap, target_spacing, stride, blur_operator=None):
    radius_voxel = [1000.0 / e / stride for e in target_spacing]
    radius_thres = [e / 3.0 for e in radius_voxel]
    # @TODO - currently, using fixed pool_ksize=3 due to performance issue
    # when using large kernel size with Pytorch (>10 sec on 224x448x448)

    # # maximum ood number which <= radius_thres
    # # 0.7071067811865475 = 1 / sqrt(2)
    # pool_ksize = int((2 * radius_thres * 0.7071067811865475 - 1) // 2 * 2 + 1)
    # # at least 3, [1,1,1] == no pooling, which increase maxRecall but significantly reduce other metrics
    # pool_ksize = max(3, pool_ksize)
    # pool_ksize = [pool_ksize, pool_ksize, pool_ksize]

    pool_ksize = [3, 3, 3]
    logger.debug(
        "Heatmap decode with radius=%s, pool_ksize=%s, radius_thres=%s",
        radius_voxel,
        pool_ksize,
        radius_thres,
    )
    del target_spacing
    assert pred_heatmap.shape[0] == 1
    ret = []
    for channel_idx, heatmap in enumerate(pred_heatmap):
        outputs = decode_heatmap_3d(
            heatmap=heatmap,
            pool_ksize=pool_ksize,
            nms_radius_thres=radius_thres,
            blur_operator=blur_operator,
            conf_thres=min(0.05, DECODE_CONF_THRES),
            max_dets=4 if (VIZ_ENABLE and MODE != "KAGGLE") else 1,
            timeout=None,
        )
        outputs = outputs.cpu()
        ret.append(outputs)
    return ret


def decode_segment_mask(
    pred_heatmap,
    target_spacing,
    stride,
    prob_thres,
    radius_factor_thres,
    conf_mode="fuse",
    prob_mode="avg",
):
    radius_voxel_thres = [
        1000 / e / stride * radius_factor_thres for e in target_spacing
    ]
    volume_thres = (
        4
        / 3
        * np.pi
        * radius_voxel_thres[0]
        * radius_voxel_thres[1]
        * radius_voxel_thres[2]
    )
    assert len(pred_heatmap.shape) == 4 and pred_heatmap.shape[0] == 1
    ret = []
    for channel_idx, heatmap in enumerate(pred_heatmap):
        keypoints = decode_segment_mask_3d(
            heatmap,
            prob_thres,
            volume_thres,
            max_dets=1,
            conf_mode=conf_mode,
            prob_mode=prob_mode,
        )
        ret.append(keypoints)
    return ret


def crop_tomo_patch(tomo, patch_position):
    # start = time.time()
    tomo_shape = tomo.shape
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
    # pad if needed
    pad = (top_pad_x, bot_pad_x, top_pad_y, bot_pad_y, top_pad_z, bot_pad_z)
    if any(pad):
        crop = F.pad(crop, pad, mode="constant", value=0)
    # end = time.time()
    # logger.debug("Crop take %f sec: %s", end - start, patch_position)
    return crop


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
    if mode != "nearest":
        ori_tomo = ori_tomo.float()

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
            size=None,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False if mode == "trilinear" else None,
            recompute_scale_factor=False,
        )[0, 0]
    if mode != "nearest":
        spaced_tomo = spaced_tomo.to(torch.uint8)  # ZYX
    spaced_tomo = spaced_tomo.cpu()
    assert spaced_tomo.dtype == torch.uint8
    return spaced_tomo


def scale_range(start, end, scale, minv, maxv):
    n = round((end - start) * scale)
    if round(end * scale) - round(start * scale) == n:
        return round(start * scale), round(end * scale)
    start1 = int(start * scale)
    start2 = round(start * scale)
    candidate_starts = []
    if minv <= start1 < start1 + n <= maxv:
        candidate_starts.append(start1)
    if minv <= start2 < start2 + n <= maxv:
        candidate_starts.append(start2)
    if len(candidate_starts) == 1:
        return candidate_starts[0], candidate_starts[0] + n
    elif len(candidate_starts) == 2:
        if candidate_starts[0] == candidate_starts[1]:
            return candidate_starts[0], candidate_starts[0] + n
        diffs = [
            abs(s - start * scale) + abs(s + n - end * scale) for s in candidate_starts
        ]
        if diffs[0] < diffs[1]:
            return candidate_starts[0], candidate_starts[0] + n
        else:
            return candidate_starts[1], candidate_starts[1] + n
    else:
        raise ValueError


@exception_printer
def tomo_producer_worker_func(
    in_queue: mp.Queue,
    out_queue: queue.Queue,
    num_remain_workers: mp.Value,
    lock: mp.Lock,
    device: str = "cpu",
    mode: str = "trilinear",
    worker_id: str = "",
):
    """Read tomo, spacing, then save to shared memory.
    Should be run at process level (multiprocessing)
    """
    logger.info("[TOMO PRODUCER %s] STARTING..", worker_id)
    _old_num_threads = torch.get_num_threads()
    if _old_num_threads < 16:
        torch.set_num_threads(16)
    logger.info(
        "[TOMO PRODUCER %s] Number of threads change from %d to %d",
        worker_id,
        _old_num_threads,
        torch.get_num_threads(),
    )
    tomo_loader = MultithreadOpencvTomogramLoader(
        num_workers=_CV2_TOMO_LOADER_NUM_THREADS
    )
    while True:
        item = in_queue.get()
        if item is None:
            with lock:
                num_remain_workers.value -= 1
                if num_remain_workers.value == 0:
                    # all other workers are stopped, this worker is the last one alive
                    # put None to indicate stopping to out_queue's consumer
                    out_queue.put(None)
            # other workers should stop too..
            in_queue.put(None)
            logger.info(
                "[TOMO PRODUCER %s] STOPPING..",
                worker_id,
            )
            return
        start = time.time()
        tomo_idx, tomo_id, ori_spacing = item
        # fetch new item
        tomo = tomo_loader.load(os.path.join(DATA_DIR, tomo_id))
        ori_shape = tomo.shape
        target_spacings = TARGET_SPACINGS
        tomo = torch.from_numpy(tomo)
        assert tomo.dtype == torch.uint8

        # Multiscale TTA
        logger.info(
            "[TOMO PRODUCER %s] Read tomo_idx=%d tomo_id=%s tomo_shape=%s",
            worker_id,
            tomo_idx,
            tomo_id,
            tomo.shape,
        )
        proxy_tomos = []
        for spacing_idx, target_spacing in enumerate(target_spacings):
            spaced_tomo = spacing_torch(
                tomo, ori_spacing, target_spacing, device=device, mode=mode
            )
            gc.collect()
            logger.info(
                "[TOMO PRODUCER %s] Spacing tomo_idx=%d tomo_id=%s shape %s -> %s",
                worker_id,
                tomo_idx,
                tomo_id,
                tomo.shape,
                spaced_tomo.shape,
            )
            proxy_tomos.append(
                ShmTensor.from_tensor(
                    spaced_tomo, f"{tomo_idx}_{tomo_id}_{spacing_idx}"
                )
            )
        del tomo
        gc.collect()
        end = time.time()
        logger.info(
            "[TOMO PRODUCER %s] Done tomo_idx=%d tomo_id=%s spacings %s -> %s with shape %s -> %s take %.2f sec",
            worker_id,
            tomo_idx,
            tomo_id,
            ori_spacing,
            target_spacings,
            ori_shape,
            tuple(e.shape for e in proxy_tomos),
            end - start,
        )
        out_queue.put((tomo_idx, tomo_id, ori_spacing, target_spacings, proxy_tomos))


@exception_printer
def patches_producer_worker_func(
    in_queue: mp.Queue,
    out_queue: queue.Queue,
    num_remain_workers: mp.Value,
    lock: threading.Lock,
    patch_size,
    border,
    overlap,
    worker_id="",
):
    """
    Should be lightweight, and expected to be run in thread level.

    Args:
        in_queue: Input queue contain item of (tomo_idx, tomo_id, ori_spacing, target_spacing, proxy_tomo)
        out_queue: Each item in output queue contains multiple patches belong to a same tomo
    """
    logger.info("[PATCHES PRODUCER %s] STARTING..", worker_id)
    while True:
        item = in_queue.get()
        if item is None:
            with lock:
                num_remain_workers.value -= 1
                if num_remain_workers.value == 0:
                    # put final None so out_queue's consumer should stop too
                    out_queue.put(None)
            # put None back to in_queue so other in_queue's consumer workers will stop too
            in_queue.put(None)
            logger.info(
                "\n\n[PATCHES PRODUCER %s] STOPPING..\n\n",
                worker_id,
            )
            return
        tomo_idx, tomo_id, ori_spacing, target_spacings, proxy_tomos = item
        start = time.time()
        assert len(proxy_tomos) == len(target_spacings) == len(TARGET_SPACINGS)

        items = []
        total_patches = 0
        cur_patch_idx = -1
        shm_tomos = []
        for target_spacing, proxy_tomo in zip(target_spacings, proxy_tomos):
            shm_tomo, tomo = proxy_tomo.to_tensor()
            tomo_shape = tomo.shape
            shm_tomos.append(shm_tomo)
            patch_positions = get_sliding_patch_positions(
                img_size=tomo.shape,
                patch_size=patch_size,
                border=border,
                overlap=overlap,
                validate=False,
            )
            # patch_positions = torch.from_numpy(patch_positions)
            patch_positions = patch_positions.tolist()
            total_patches += len(patch_positions)

            for patch_pos in patch_positions:
                # crop tomo patch
                crop = crop_tomo_patch(tomo, patch_pos)
                crop = crop[None]  # 1ZYX, uint8
                cur_patch_idx += 1
                item = {
                    "tomo_idx": torch.tensor(tomo_idx, dtype=torch.int16),
                    "patch_pos": torch.tensor(patch_pos, dtype=torch.float32),
                    "tomo_shape": torch.tensor(tomo_shape, dtype=torch.int16),
                    "ori_spacing": torch.tensor(ori_spacing, dtype=torch.float32),
                    "target_spacing": torch.tensor(target_spacing, dtype=torch.float32),
                    "cur_patch_idx": torch.tensor(cur_patch_idx, dtype=torch.int16),
                    "image": crop,
                }
                items.append(item)
        assert cur_patch_idx == total_patches - 1
        for item in items:
            item["total_patches"] = torch.tensor(total_patches, dtype=torch.int16)

        end = time.time()
        logger.debug(
            "[PATCHES PRODUCER %s] Done tomo_idx=%d tomo_id=%s take %.2f sec",
            worker_id,
            tomo_idx,
            tomo_id,
            end - start,
        )
        out_queue.put((tomo_idx, shm_tomos, items))


@exception_printer
def batch_producer_worker_func(
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    batch_size,
    device="cpu",
):
    """Should be lightweight, and expected to be run in thread level."""
    count = 0
    items = []
    should_continue = True
    shm_dict = {}
    last_tomo_idx = -1
    while should_continue:
        item = in_queue.get()
        if item is None:
            # put None so in_queue's consumer should notice and stop too
            in_queue.put(None)
            should_continue = False
        else:
            tomo_idx, shm_tomos, new_items = item
            # @TODO - remove this line
            assert all([tomo_idx == e["tomo_idx"].item() for e in new_items])
            items.extend(new_items)
            shm_dict[tomo_idx] = shm_tomos

        while True:
            if len(items) == 0 or (len(items) < batch_size and should_continue):
                break
            start = time.time()
            cur_bs = min(batch_size, len(items))
            batch = items[:cur_bs]
            batch = default_collate(batch)
            if device != "cpu":
                batch = {k: v.to(device) for k, v in batch.items()}
            assert batch["image"].dtype == torch.uint8
            end = time.time()
            logger.debug(
                "[BATCH PRODUCER] new batch %d (tomo %d) take %.4f",
                count,
                tomo_idx,
                end - start,
            )
            out_queue.put(batch)
            count += 1
            last_tomo_idx = batch["tomo_idx"][-1].item()
            items = items[cur_bs:]
        for shm_tomo_idx in list(shm_dict.keys()):
            # un-collated batch should start with tomo_idx >= last_tomo_idx
            # so we're safe to unlink previous one
            # since default_collate() copy to a newly allocated tensor
            if shm_tomo_idx < last_tomo_idx:
                for shm in shm_dict[shm_tomo_idx]:
                    shm.unlink()
                del shm_dict[shm_tomo_idx]

    for shms in shm_dict.values():
        for shm in shms:
            shm.unlink()
    del shm_dict
    gc.collect()
    # put a final None to indicate `all done`, stop inference worker too..
    out_queue.put(None)
    logger.info("[BATCH PRODUCER] Stopped since tomo_idx=%d", tomo_idx)
    assert len(items) == 0


@exception_printer
def inference_worker(
    worker_id, input_queue, gpu_id, sigs, names, weights=None, heatmap_ensemble=True
):
    logger.info(
        "STARTING INFERENCE WORKER %d with device cuda:%d, current input queue size %d, %d models with names %s",
        worker_id,
        gpu_id,
        input_queue.qsize(),
        len(sigs),
        names,
    )
    assert isinstance(weights, list) and len(weights) == len(sigs)
    if heatmap_ensemble:
        assert len(names) == 1
    else:
        assert len(names) == len(sigs)

    device = torch.device(f"cuda:{gpu_id}")
    patch_size = sigs[0]["patch_size"]
    border = sigs[0]["border"]
    overlap = sigs[0]["overlap"]

    ############ START ALL WORKERS ###########
    ##########################################
    tomo_queue = mp.Queue(maxsize=TOMO_PRODUCER_PREFETCH)
    tomo_producer_num_remain_workers = mp.Value("i", TOMO_PRODUCER_NUM_WORKERS)
    tomo_producer_lock = mp.Lock()
    patches_queue = queue.Queue(maxsize=PATCHES_PRODUCER_PREFETCH)
    patches_producer_num_remain_workers = mp.Value("i", PATCHES_PRODUCER_NUM_WORKERS)
    patches_producer_lock = threading.Lock()
    batch_queue = queue.Queue(maxsize=DATALOADER_PREFETCH)

    # PRODUCE SPACED TOMO
    tomo_producer_workers = []
    for _worker_id in range(TOMO_PRODUCER_NUM_WORKERS):
        worker = mp.Process(
            group=None,
            target=tomo_producer_worker_func,
            name=f"tomo_producer_{worker_id}.{_worker_id}",
            args=(
                input_queue,
                tomo_queue,
                tomo_producer_num_remain_workers,
                tomo_producer_lock,
            ),
            kwargs={
                "device": TOMO_SPACING_DEVICE,
                "mode": TOMO_SPACING_MODE,
                "worker_id": f"{worker_id}.{_worker_id}",
            },
        )
        worker.start()
        tomo_producer_workers.append(worker)

    # PRODUCER TOMO PATCHES
    patches_producer_workers = []
    for _worker_id in range(PATCHES_PRODUCER_NUM_WORKERS):
        worker = threading.Thread(
            group=None,
            target=patches_producer_worker_func,
            name=f"patches_producer_{worker_id}.{_worker_id}",
            args=(
                tomo_queue,
                patches_queue,
                patches_producer_num_remain_workers,
                patches_producer_lock,
                patch_size,
                border,
                overlap,
            ),
            kwargs={"worker_id": f"{worker_id}.{_worker_id}"},
        )
        worker.start()
        patches_producer_workers.append(worker)

    # BATCH PRODUCER
    batch_producer_worker = threading.Thread(
        group=None,
        target=batch_producer_worker_func,
        name="batch_producer",
        args=(patches_queue, batch_queue, BATCH_SIZE),
        kwargs={"device": "cuda:0"},
    )
    batch_producer_worker.start()

    ############## LOAD MODEL ##################
    ############################################
    logger.info("LOADING MODEL..")
    models = []
    for sig in sigs:
        # ensure all model use same patch size
        assert (
            sig["patch_size"] == patch_size
            and sig["border"] == border
            and sig["overlap"] == overlap
        )
        model = KFoldModelWrapper(
            sig,
            base_dir=ASSETS_DIR,
            load_combined_weight=LOAD_COMBINED_WEIGHT,
            heatmap_agg_mode=HEATMAP_AGG_MODE,
            save_memory=FORWARD_SAVE_MEMORY,
            act="identity" if HEATMAP_AGG_LOGITS else "sigmoid",
        )
        if MODE == "LOCAL" and worker_id == 0:
            # SAVE COMBINED TORCH WEIGHT
            torch_combined_weight_path = os.path.join(
                ASSETS_DIR, sig["torch_combined_weight_path"]
            )
            if not os.path.isfile(torch_combined_weight_path):
                logger.warning(
                    f"MODE=%s, saving combined weight to %s",
                    MODE,
                    torch_combined_weight_path,
                )
                torch.save(model.state_dict(), torch_combined_weight_path)
        model = TTAModelWrapper(
            model,
            sig["tta"],
            heatmap_agg_mode=HEATMAP_AGG_MODE,
            save_memory=FORWARD_SAVE_MEMORY,
        )
        model.eval().to(device)
        models.append(model)
    num_models = len(models)
    if not heatmap_ensemble:
        assert len(models) == len(names)
    logger.info("-----------\nLOADED %d MODELS:\n%s------------", len(models), models)

    # Postprocess: heatmap blurring
    if DECODE_NMS_BLUR_SIGMA is not None:
        from yagm.transforms import monai_custom as CT

        sigma_voxel = [
            1000 / e / HEATMAP_STRIDE * DECODE_NMS_BLUR_SIGMA
            for e in TARGET_SPACINGS[0]
        ]
        blur_operator = (
            CT.CustomGaussianFilter(
                spatial_dims=3,
                sigma=sigma_voxel,
                truncated=4,
                approx="erf",
                requires_grad=False,
            )
            .eval()
            .to(device)
        )
    else:
        blur_operator = None
    logger.info(
        "POST-PROCESSING BLUR OPERATOR:\n%s on device %s",
        blur_operator,
        getattr(blur_operator, "device", None),
    )

    if heatmap_ensemble:
        all_cache_keys = [0]
    else:
        all_cache_keys = list(range(num_models))
    cache_results = {
        k: {
            "cur_pred_heatmap_sum": None,
            "cur_pred_heatmap_count": None,
            "cur_tomo_idx": None,
            "cur_tomo_id": None,
            "main_target_spacing": None,
            "submission": SubmissionDataFrame(),
        }
        for k in all_cache_keys
    }
    logger.info("\n\n\nWORKER %d STARTING INFERENCE..\n", worker_id)
    infer_start = time.time()
    cur_batch_idx = 0
    pbar = tqdm(desc=f"===== INFERENCE {worker_id} =====")
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        while True:
            batch = batch_queue.get()
            if batch is None:
                break
            # print("GOT BATCH:", {(k, v.shape, v.dtype) for k, v in batch.items()})
            batch_image = batch["image"].to(device).float()
            B = batch_image.shape[0]
            batch_heatmaps = [model(batch_image) for model in models]
            assert len(batch_heatmaps) == len(models)

            for i in range(B):
                for model_idx in range(num_models):
                    weight = weights[model_idx]
                    pred_patch_heatmap = batch_heatmaps[model_idx][i]
                    C = pred_patch_heatmap.shape[0]
                    assert C == 1

                    tomo_idx = batch["tomo_idx"][i]
                    tomo_id = ALL_TOMO_IDS[tomo_idx]
                    tomo_shape = batch["tomo_shape"][i].tolist()
                    ori_spacing = batch["ori_spacing"][i].tolist()
                    target_spacing = batch["target_spacing"][i].tolist()
                    cur_patch_idx = int(batch["cur_patch_idx"][i].item())
                    total_patches = int(batch["total_patches"][i].item())
                    patch_position = batch["patch_pos"][i].cpu()

                    if heatmap_ensemble:
                        # multiple models share same first/last patch concept
                        is_first = cur_patch_idx == 0 and model_idx == 0
                        is_last = (
                            cur_patch_idx == total_patches - 1
                            and model_idx == num_models - 1
                        )
                    else:
                        # each model maintain its own aggregated heatmap and first/last patch concept
                        is_first = cur_patch_idx == 0
                        is_last = cur_patch_idx == total_patches - 1

                    # load cache for this tomo+model combination
                    cache_key = 0 if heatmap_ensemble else model_idx
                    model_result = cache_results[cache_key]
                    ### start IS_FIRST
                    if is_first:
                        logger.info(
                            "[INFER %d] Model %s receive first patch of tomo %s (index %d)",
                            worker_id,
                            model_idx,
                            tomo_id,
                            tomo_idx,
                        )
                        assert all(
                            [
                                model_result[k] is None
                                for k in [
                                    "cur_tomo_idx",
                                    "cur_tomo_id",
                                    "cur_pred_heatmap_sum",
                                    "cur_pred_heatmap_count",
                                    "main_target_spacing",
                                ]
                            ]
                        )

                        # allocate new heatmap_sum and heatmap_count
                        model_result["cur_tomo_idx"] = tomo_idx
                        model_result["cur_tomo_id"] = tomo_id
                        shared_agg_heatmap_shape = [
                            round(e / HEATMAP_STRIDE) for e in tomo_shape
                        ]
                        # float16 to save memory
                        model_result["cur_pred_heatmap_sum"] = torch.zeros(
                            (C, *shared_agg_heatmap_shape),
                            dtype=torch.float16,
                            device=pred_patch_heatmap.device,
                        )
                        model_result["cur_pred_heatmap_count"] = torch.zeros(
                            (C, *shared_agg_heatmap_shape),
                            dtype=torch.float16,
                            device=pred_patch_heatmap.device,
                        )
                        model_result["main_target_spacing"] = target_spacing
                        assert all(
                            [
                                abs(a - b) < 1e-4
                                for a, b in zip(target_spacing, TARGET_SPACINGS[0])
                            ]
                        )
                    ### end IS_FIRST

                    assert tomo_idx == model_result["cur_tomo_idx"]
                    assert all(
                        [
                            a <= b
                            for a, b in zip(
                                model_result["main_target_spacing"], target_spacing
                            )
                        ]
                    )
                    cur_pred_heatmap_sum = model_result["cur_pred_heatmap_sum"]
                    cur_pred_heatmap_count = model_result["cur_pred_heatmap_count"]
                    main_target_spacing = model_result["main_target_spacing"]
                    cur_submission: SubmissionDataFrame = model_result["submission"]
                    shared_agg_heatmap_shape = cur_pred_heatmap_sum.shape[1:]

                    scale_zyx = [
                        a / b for a, b in zip(shared_agg_heatmap_shape, tomo_shape)
                    ]
                    scale_z, scale_y, scale_x = scale_zyx

                    # interpolate to shared coordinate space
                    C, Z2, Y2, X2 = pred_patch_heatmap.shape
                    assert C == 1
                    patch_heatmap_shape = [
                        round(e * s) for e, s in zip(patch_size, scale_zyx)
                    ]
                    if [Z2, Y2, X2] != patch_heatmap_shape:
                        logger.debug(
                            "PATCH HEATMAP INTERPOLATE: %s --> %s",
                            (Z2, Y2, X2),
                            patch_heatmap_shape,
                        )
                        pred_patch_heatmap = F.interpolate(
                            pred_patch_heatmap[None],  # (1, C, Z2, Y2, X2)
                            size=patch_heatmap_shape,
                            mode=HEATMAP_INTERPOLATION_MODE,
                            align_corners=False,
                        )[
                            0
                        ]  # (C, Z3, Y3, X3)
                    else:
                        logger.debug("PATCH HEATMAP SAME: %s", pred_patch_heatmap.shape)
                        pass

                    roi_start, roi_end, patch_start, patch_end = patch_position.tolist()
                    top_pad_z, top_pad_y, top_pad_x = [
                        max(-start, 0) for start in roi_start
                    ]
                    bot_pad_z, bot_pad_y, bot_pad_x = [
                        max(0, end - size) for end, size in zip(roi_end, tomo_shape)
                    ]
                    assert (
                        tuple(
                            round((pe - ps) * s)
                            for pe, ps, s in zip(patch_end, patch_start, scale_zyx)
                        )
                        == pred_patch_heatmap.shape[1:]
                    )
                    sz, sy, sx = tuple(
                        rs - ps for rs, ps in zip(roi_start, patch_start)
                    )
                    ez, ey, ex = tuple(
                        ps - ce + pe
                        for ps, ce, pe in zip(patch_size, patch_end, roi_end)
                    )

                    assert (roi_end[0] - bot_pad_z) - (roi_start[0] + top_pad_z) == (
                        ez - bot_pad_z
                    ) - (sz + top_pad_z)
                    assert (roi_end[1] - bot_pad_y) - (roi_start[1] + top_pad_y) == (
                        ey - bot_pad_y
                    ) - (sy + top_pad_y)
                    assert (roi_end[2] - bot_pad_x) - (roi_start[2] + top_pad_x) == (
                        ex - bot_pad_x
                    ) - (sx + top_pad_x)

                    dst_shape = shared_agg_heatmap_shape
                    dst_slices = [
                        slice(None),
                        slice(
                            *scale_range(
                                roi_start[0] + top_pad_z,
                                roi_end[0] - bot_pad_z,
                                scale_z,
                                0,
                                dst_shape[0],
                            )
                        ),
                        slice(
                            *scale_range(
                                roi_start[1] + top_pad_y,
                                roi_end[1] - bot_pad_y,
                                scale_y,
                                0,
                                dst_shape[1],
                            )
                        ),
                        slice(
                            *scale_range(
                                roi_start[2] + top_pad_x,
                                roi_end[2] - bot_pad_x,
                                scale_x,
                                0,
                                dst_shape[2],
                            )
                        ),
                    ]
                    src_shape = pred_patch_heatmap.shape[1:]
                    src_slices = [
                        slice(None),
                        slice(
                            *scale_range(
                                sz + top_pad_z, ez - bot_pad_z, scale_z, 0, src_shape[0]
                            )
                        ),
                        slice(
                            *scale_range(
                                sy + top_pad_y, ey - bot_pad_y, scale_y, 0, src_shape[1]
                            )
                        ),
                        slice(
                            *scale_range(
                                sx + top_pad_x, ex - bot_pad_x, scale_x, 0, src_shape[2]
                            )
                        ),
                    ]

                    cur_pred_heatmap_sum[dst_slices] += (
                        pred_patch_heatmap[src_slices] * weight
                    )
                    cur_pred_heatmap_count[dst_slices] += weight

                    if is_last:
                        # DECODE HEATMAP TO COORDINATE
                        tomo_id = model_result["cur_tomo_id"]
                        logger.info(
                            "[INFER %d] Receive last patch of tomo %s (index %d), decoding..",
                            worker_id,
                            tomo_id,
                            tomo_idx,
                        )

                        # ensure prediction cover all the tomogram
                        assert torch.all(cur_pred_heatmap_count > 0)
                        heatmap = torch.div(
                            cur_pred_heatmap_sum,
                            cur_pred_heatmap_count,
                            out=cur_pred_heatmap_sum,
                        )
                        if HEATMAP_AGG_LOGITS:
                            heatmap = F.sigmoid(heatmap)
                        # Decode heatmap
                        _start = time.time()
                        if DECODE_METHOD == "nms":
                            outputs = decode_heatmap(
                                heatmap,
                                main_target_spacing,
                                HEATMAP_STRIDE,
                                blur_operator=blur_operator,
                            )
                        elif DECODE_METHOD == "cc3d":
                            outputs = decode_segment_mask(
                                heatmap,
                                main_target_spacing,
                                HEATMAP_STRIDE,
                                prob_thres=DECODE_CC3D_CONF_THRES,
                                radius_factor_thres=DECODE_CC3D_RADIUS_FACTOR,
                                conf_mode=DECODE_CC3D_CONF_MODE,
                                prob_mode=DECODE_CC3D_PROB_MODE,
                            )
                        else:
                            raise ValueError
                        _end = time.time()
                        logger.debug(
                            "Decode heatmap of shape %s take %.4f sec",
                            heatmap.shape,
                            _end - _start,
                        )
                        # Convert to submission coordinates
                        assert len(outputs) == 1
                        main_target_spacing_tensor = torch.tensor(
                            [main_target_spacing], dtype=torch.float32
                        )  # (1, 3)
                        ori_spacing_tensor = torch.tensor(
                            [ori_spacing], dtype=torch.float32
                        )  # (1, 3)
                        for _channel_idx, keypoints in enumerate(outputs):
                            # back to original coordinate space
                            keypoints[:, :3] = (keypoints[:, :3] + 0.5) * (
                                HEATMAP_STRIDE
                                * main_target_spacing_tensor
                                / ori_spacing_tensor
                            ) - 0.5
                            keypoints = keypoints.tolist()

                            ###### VISUALIZATION ######
                            if VIZ_ENABLE and MODE != "KAGGLE":
                                tomo_dir = os.path.join(DATA_DIR, tomo_id)
                                row = GT_DF[GT_DF["tomo_id"] == tomo_id].iloc[0]
                                ori_shape = (row["Z"], row["Y"], row["X"])
                                gt_zyx = [
                                    row["motor_z"],
                                    row["motor_y"],
                                    row["motor_x"],
                                ]
                                viz = viz_byu(
                                    tomo_dir,
                                    ori_spacing,
                                    ori_shape,
                                    gt_zyx,
                                    heatmap[0],
                                    keypoints,
                                )
                                # get the kind/judge of current prediction, one of TP, TN, FP, FN
                                if (
                                    len(keypoints) > 0
                                    and keypoints[0][3] >= DECODE_CONF_THRES
                                ):
                                    # predicted as positive
                                    pred_zyx = keypoints[0][:3]
                                    if gt_zyx[0] == -1:
                                        # GT is negative
                                        judge = "FP"
                                    else:
                                        dist = (
                                            (gt_zyx[0] - pred_zyx[0]) ** 2
                                            + (gt_zyx[1] - pred_zyx[1]) ** 2
                                            + (gt_zyx[2] - pred_zyx[2]) ** 2
                                        ) ** 0.5
                                        if dist <= 1000 / ori_spacing:
                                            judge = "TP"
                                        else:
                                            judge = "FN"
                                else:
                                    # predicted as negative
                                    if gt_zyx[0] == -1:
                                        # GT is negative
                                        judge = "TN"
                                    else:
                                        judge = "FN"

                                save_name = f"{judge}-{tomo_idx}-{tomo_id}-{'x'.join([str(int(e)) for e in ori_shape])}-spacing{ori_spacing}.jpg"
                                save_path = os.path.join(TMP_VIZ_DIR, judge, save_name)
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                cv2.imwrite(
                                    save_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
                                )

                            if len(keypoints) > 0:
                                # get top-1 prediction only
                                z, y, x, conf = keypoints[0]
                                # one motor detected
                                cur_submission.add_row(tomo_id, x, y, z, conf)
                            else:
                                # no motor detected
                                cur_submission.add_row(tomo_id, -1, -1, -1, 0.0)

                        # clear the cache
                        for k in [
                            "cur_tomo_idx",
                            "cur_tomo_id",
                            "cur_pred_heatmap_sum",
                            "cur_pred_heatmap_count",
                            "main_target_spacing",
                        ]:
                            model_result[k] = None
                        gc.collect()
                        # torch.cuda.empty_cache()
            cur_batch_idx += 1
            pbar.update(1)
    pbar.close()
    infer_end = time.time()

    logger.info(
        "\n\n\n INFER WORKER %d INFER ON %d BATCHES TAKE %.4f sec",
        worker_id,
        cur_batch_idx,
        infer_end - infer_start,
    )

    if heatmap_ensemble:
        assert len(cache_results) == 1 and list(cache_results.keys())[0] == 0
    for model_idx, model_result in cache_results.items():
        submision = model_result["submission"]
        df = submision.to_pandas(submit=False)
        name = names[model_idx]
        csv_path = os.path.join(TMP_CSV_DIR, f"{name}.csv")
        df.to_csv(csv_path, index=False)

    # JOIN
    logger.info("Worker %d waiting for all worker to be finished..", worker_id)
    all_workers = [
        batch_producer_worker,
        *tomo_producer_workers,
        *patches_producer_workers,
    ]
    for worker in all_workers:
        worker.join()
    return None


########################### MAIN CODE #############################
###################################################################
logger.warning("CLEARING ALL CONTENTS IN: %s", [TMP_CSV_DIR, TMP_CSV_DIR, TMP_VIZ_DIR])
clear_dir_content("/dev/shm/")
clear_dir_content(TMP_CSV_DIR)
clear_dir_content(TMP_VIZ_DIR)
if MODE == "LOCAL":
    clear_dir_content(WORKING_DIR)
os.makedirs(TMP_CSV_DIR, exist_ok=True)
os.makedirs(TMP_VIZ_DIR, exist_ok=True)


global_start = time.time()
all_names = []
for job_idx, job in enumerate(JOBS):
    input_queue = mp.Queue()
    for tomo_idx, (tomo_id, ori_spacing) in enumerate(
        zip(ALL_TOMO_IDS, ALL_TOMO_SPACINGS)
    ):
        input_queue.put((tomo_idx, tomo_id, [ori_spacing] * 3))
    # None to indicate ending and workers should stop when receive this None
    input_queue.put(None)

    job_workers = []
    for part_idx, device_id in enumerate(DEVICES):
        names = [f"{main_name}_part{part_idx}" for main_name in job["names"]]
        all_names.extend(names)
        assert len(all_names) == len(
            set(all_names)
        ), "Result name must be unique to each other"
        worker = mp.Process(
            group=None,
            target=inference_worker,
            name=f"INFERENCE_WORKER_job={job_idx}_part={part_idx}",
            kwargs={
                "worker_id": part_idx,
                "input_queue": input_queue,
                "gpu_id": device_id,
                "sigs": job["sigs"],
                "names": names,
                "weights": job["weights"],
                "heatmap_ensemble": HEATMAP_ENSEMBLE,
            },
        )
        worker.start()
        job_workers.append(worker)

    for worker in job_workers:
        worker.join()
global_end = time.time()
logger.info(
    "\n\n\n>>>>>> FINISH ALL INFERENCE TASK WITHIN %.4f sec (%.4f min)",
    global_end - global_start,
    (global_end - global_start) / 60.0,
)


dfs = []
for name in all_names:
    csv_path = os.path.join(TMP_CSV_DIR, f"{name}.csv")
    try:
        df = pd.read_csv(csv_path)
        dfs.append(df)
        logger.info(
            "Append new temporary prediction Dataframe with shape %s, columns=%s",
            df.shape,
            list(df.columns),
        )
    except Exception as e:
        logger.warning(
            "EXCEPTION OCCUR when processing name=%s:\n%s\nIGNORE CSV RESULT FILE AT %s",
            name,
            e,
            csv_path,
        )
        pass

df: pd.DataFrame = pd.concat(dfs, axis=0, ignore_index=True).reset_index(drop=True)
logger.info(
    "CONCAT ALL TEMPORARY DATAFRAMES INTO A SINGLE DATAFRAME WITH SHAPE %s, COLUMNS=%s\n%s",
    df.shape,
    list(df.columns),
    df,
)

if MODE in ["LOCAL", "KAGGLE_VAL"]:
    assert (
        len(GT_DF) == len(df) == len(ALL_TOMO_IDS)
    ), f"{GT_DF.shape} {df.shape} {len(ALL_TOMO_IDS)}"
    metrics = compute_metrics(GT_DF, df, mode="kaggle")
    logger.info("%s METRICS:\n%s", MODE, metrics)

# Filter based on conf
if QUANTILE_THRES is None:
    filter_thres = DECODE_CONF_THRES
else:
    assert 0 <= QUANTILE_THRES <= 1.0
    pred_confs = df["conf"].values
    filter_thres = np.quantile(pred_confs, QUANTILE_THRES)
    assert 0 <= filter_thres <= 1.0


print("USING FILTER THRESHOLD:", filter_thres)
df.loc[df["conf"] < filter_thres, ["motor_z", "motor_y", "motor_x"]] = -1
if MODE in ["LOCAL", "KAGGLE_VAL"]:
    kaggle_fbeta, kaggle_precision, kaggle_recall, (tn, fp, fn, tp) = kaggle_score(
        GT_DF, df
    )
    logger.info(
        "--------------\nKAGGLE METRICS AT THRESHOLD=%f:\n\nFBETA=%f PRECISION=%f RECALL=%f\nTN=%d FP=%d FN=%d TP=%d--------------\n\n",
        filter_thres,
        kaggle_fbeta,
        kaggle_precision,
        kaggle_recall,
        tn,
        fp,
        fn,
        tp,
    )

df.rename(
    columns={
        "motor_z": "Motor axis 0",
        "motor_y": "Motor axis 1",
        "motor_x": "Motor axis 2",
    },
    inplace=True,
)
df = df[["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"]]
logger.info("SUBMISSION DATAFRAME:\n%s", df)

if MODE == "KAGGLE":
    logger.warning(f"CLEARING ALL CONTENTS IN %s", WORKING_DIR)
    clear_dir_content(WORKING_DIR)

submission_csv_path = os.path.join(WORKING_DIR, "submission.csv")
df.to_csv(submission_csv_path, index=False)
