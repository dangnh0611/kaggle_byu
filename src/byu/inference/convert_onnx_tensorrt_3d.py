import gc
import logging
import os
import random

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from yagm.utils import lightning as l_utils
from yagm.utils.torch2onnx import onnx2trt, torch2onnx

from byu.data.datasets.heatmap_3d_dataset import Heatmap3dDataset
from byu.data.io import OpencvTomogramLoader

logging.basicConfig(level=logging.INFO)

# DATA_DIR = "/home/dangnh36/datasets/.comp/byu/"
# DATASET_3D_CONFIG = f"""
# env:
#     data_dir: {DATA_DIR}
# cv:
#     strategy: skf4_rd42
#     num_folds: 4
#     fold_idx: 0
#     train_on_all: False
# loader:
#     num_workers: 1
# data:
#     label_fname: gt_v2  # gt | gt_v2 | gt_v3 | all_gt | all_gt_v3
#     patch_size: [224, 448, 448]
#     start: [0, 0, 0]
#     overlap: [0, 0, 0]
#     border: [0, 0, 0]

#     sigma: 0.2
#     fast_val_workers: 1
#     fast_val_prefetch: 1
#     io_backend: cv2  # cv2, cv2_seq, npy, cache
#     crop_outside: True
#     ensure_fg: False
#     label_smooth: [0.0,1.0]
#     filter_rule: null  # null | eq1 | le1

#     sampling:
#         method: pre_patch  # pre_patch | rand_crop
#         pre_patch:
#             fg_max_dup: 1
#             bg_ratio: 0.0
#             bg_from_pos_ratio: 0.01
#             overlap: [0, 0, 0]
#         rand_crop:
#             random_center: True
#             margin: 0.25
#             auto_correct_center: True
#             pos_weight: 1
#             neg_weight: 1
#     transform:
#         resample_mode: trilinear # F.grid_sample() mode
#         target_spacing: [16,16,16]

#         heatmap_mode: gaussian
#         heatmap_stride: [1,1,1]
#         heatmap_same_sigma: False
#         heatmap_same_std: False
#         lazy: True
#         device: null
#     aug:
#         enable: True
#         zoom_prob: 0.4
#         zoom_range: [[0.6, 1.2], [0.6, 1.2], [0.6, 1.2]]  # (X, Y, Z) or (H, W, D)
#         # affine1
#         affine1_prob: 0.5
#         affine1_scale: 0.3  # max_skew_xy = 1.3 / 0.7 = 1.86
#         # affine2
#         affine2_prob: 0.25
#         affine2_rotate_xy: 15 # degrees
#         affine2_scale: 0.3  # max_skew_xy = 1.3 / 0.7 = 1.86
#         affine2_shear: 0.2

#         rand_shift: False  # only used in rand_crop, very useful if random_center=auto_correct_center=False

#         # no lazy, can't properly transform points
#         grid_distort_prob: 0.0
#         smooth_deform_prob: 0.0

#         intensity_prob: 0.5
#         smooth_prob: 0.0
#         hist_equalize: False
#         downsample_prob: 0.2
#         coarse_dropout_prob: 0.1

#         # MIXER
#         mixup_prob: 0.0
#         cutmix_prob: 0.0
#         mixer_alpha: 1.0
#         mixup_target_mode: max
#     tta:
#         # enable: [zyx]
#         enable: [zyx, zxy, zyx_x, zyx_y]
# """


# global_cfg = OmegaConf.create(DATASET_3D_CONFIG)
# dataset = Heatmap3dDataset(global_cfg, stage="train")
# if dataset.stage != "train":
#     dataset.fast_val_tomo_loader.start()

# imgs = []
# random.seed(42)
# # select_idxs = random.choices(list(range(len(dataset))), k = 50)
# select_idxs = list(range(1))
# print("SELECT RANDOM INDICES:", select_idxs)
# for idx in select_idxs:
#     sample = dataset[idx]
#     img = sample["image"]
#     assert img.dtype == torch.uint8
#     print(idx, img.shape, img.dtype)
#     imgs.append(img[None].float().contiguous())
# del dataset
# gc.collect()


tomo_loader = OpencvTomogramLoader()
# TOMO_ROOT_DIR = "/home/dangnh36/datasets/.comp/byu/raw/train/"
TOMO_ROOT_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/"
tomo = tomo_loader.load(os.path.join(TOMO_ROOT_DIR, "tomo_bfd5ea"))
tomo = (
    torch.from_numpy(tomo[None, None, 0:224, 0:448, 0:448]).float().contiguous()
)  # (1, 1, Z, Y, X)
imgs = [tomo]


############### LOAD MODEL ###############
DEVICE = "cuda:0"
CONVERT_ONNX = True
CONVERT_TRT = True


CONFIG_PATH = "assets/EXP31_RESNEXT50_ALLGTV3_config.yaml"
WEIGHT_PATH = "assets/EXP31_RESNEXT50_ALLGTV3_ep1_step32000.ckpt"
TRT_SAVE_PATH = "/kaggle/working/EXP31_RESNEXT50_ALLGTV3_ep1_step32000.engine"
WORKSPACE_SIZE = int(5 * (2**30))
PRECISION = "fp16"


ONNX_SAVE_PATH = None

MIN_BS, OPT_BS, MAX_BS = 1, 1, 1

if TRT_SAVE_PATH is None:
    TRT_SAVE_PATH = WEIGHT_PATH.replace(".ckpt", ".engine")

if ONNX_SAVE_PATH is None:
    ONNX_SAVE_PATH = TRT_SAVE_PATH.replace(".engine", ".onnx")

os.makedirs(os.path.dirname(ONNX_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(TRT_SAVE_PATH), exist_ok=True)


if CONVERT_ONNX:
    device = torch.device(DEVICE)
    cfg = OmegaConf.load(CONFIG_PATH)

    ######### OVERWRITE SOME CONFIGS ##########
    cfg.misc.log_model = False
    print("EMA VAL DECAYS:", cfg.ema.val_decays)
    cfg.model.head.ms = [0]
    cfg.ckpt.strict = True
    if "x3d" in cfg.model.encoder._target_ or "i3d" in cfg.model.encoder._target_:
        cfg.model.encoder.pretrained = None
    elif "smp" in cfg.model.encoder._target_:
        cfg.model.encoder.weights = None
        if "resnet101" in cfg.model.encoder.model_name:
            cfg.model.encoder.model_name = "resnet101"
    else:
        pass
    # for 2.5D encoder
    try:
        cfg.model.encoder.encoder_2d.pretrained = False
    except:
        pass
    try:
        if cfg.model.encoder.encoder_2d.model_name == "convnext_nano.r384_ad_in12k":
            cfg.model.encoder.encoder_2d.model_name = "convnext_nano"
    except:
        pass
    ##########

    task = l_utils.build_task(cfg)
    print(task.device)
    l_utils.load_lightning_state_dict(
        model=task,
        ckpt_path=WEIGHT_PATH,
        cfg=cfg,
    )
    print(f"Loaded state dict")

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self._model = model

        def forward(self, x):
            ret = self._model(x)
            assert isinstance(ret, list) and len(ret) == 1
            return ret[0]

    torch_model = ModelWrapper(task.model)
    print(torch_model)
    torch_model.eval().to(device)
    print("DEVICE:", device)

    torch2onnx(
        torch_model=torch_model,
        sample_inputs=[imgs[0]],
        input_names=["image"],
        output_names=["heatmap"],
        save_path=ONNX_SAVE_PATH,
        precision="fp32",
        device=DEVICE,
        dynamic_batching=True,
        batch_axis=0,
        validate_fn=None,
        rtol=1e-3,
        atol=1e-4,
        verbose=True,
    )
    del torch_model
    gc.collect()
    torch.cuda.empty_cache()


if CONVERT_TRT:
    try:
        engine = onnx2trt(
            onnx_file_path=ONNX_SAVE_PATH,
            engine_file_path=TRT_SAVE_PATH,
            input_shape=imgs[0].shape[1:],
            precision=PRECISION,
            min_batch=MIN_BS,
            opt_batch=OPT_BS,
            max_batch=MAX_BS,
            enable_dynamic_batching=True,
            workspace_size=WORKSPACE_SIZE,
        )

        if engine is not None:
            print("Conversion successful!")
        else:
            print("Conversion failed!")
    except Exception as e:
        print(f"Conversion error: {e}")
