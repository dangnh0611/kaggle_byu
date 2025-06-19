import warnings

warnings.simplefilter("ignore")
import json
import logging
import os
import shutil
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from yagm.transforms.keypoints.decode import decode_heatmap_3d, decode_segment_mask_3d
from yagm.utils import hydra as hydra_utils

# Setting up custom logging
from byu.utils.data import SubmissionDataFrame
from byu.utils.metrics import compute_metrics, kaggle_score
from byu.utils.wbf import weighted_boxes_fusion_3d

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


ALL_UNIQUE_AGG_NAMES = [
    "DENSENET121-zyx_x",
    "DENSENET121-zyx_y",
    "RESNEXT50-zxy_z",
    "RESNEXT50-zyx_z",
    "X3D-zxy",
    "X3D-zyx",
]
ALL_UNIQUE_AGG_WEIGHTS = {agg_name: 1.0 for agg_name in ALL_UNIQUE_AGG_NAMES}


##################################################
###### GLOBAL VARS
##################################################
_CV2_TOMO_LOADER_NUM_THREADS = 8

PATCH_SIZE = [224, 448, 448]
PATCH_BORDER = [0, 0, 0]
PATCH_OVERLAP = [0, 0, 0]

IO_BACKEND = "cv2"
BASE_SPACING = 13.1  # avg 15.6
# TARGET_SPACINGS = [13.1, 16.0, 19.7]
TARGET_SPACINGS = [[16.0, 16.0, 16.0]]
TOMO_SPACING_DEVICE = "cpu"
TOMO_SPACING_MODE = "trilinear"

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
DECODE_CONF_THRES = 0.05
QUANTILE_THRES = 0.55

SAVE_HEATMAP_NPY = False
DECODE_HEATMAP_TO_CSV = True
CSV_ENSEMBLE_MODE = "wbf"  # max | wbf
WBF_CONF_TYPE = "avg"  # avg | max
WBF_CONF_THRES = 0.2

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
    DEVICES = [0]
    ASSETS_DIR = "assets/"
    WORKING_DIR = "outputs/submit/working_backup2/"

    DATA_DIR = "/home/dangnh36/datasets/.comp/byu/raw/train"
    VAL_GT_CSV_PATH = "/home/dangnh36/datasets/.comp/byu/processed/gt_v3.csv"
    with open(
        "/home/dangnh36/datasets/.comp/byu/processed/cv/v3/skf4_rd42.json", "r"
    ) as f:
        cv_meta = json.load(f)
        ALL_TOMO_IDS = cv_meta["folds"][0]["val"][:]  # fold 0

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
SAVE_HEATMAP_NPY_DIR = os.path.join(WORKING_DIR, "tmp_heatmap")

# RECHECK SOME CONDITIONS
assert len(ALL_TOMO_IDS) == len(ALL_TOMO_SPACINGS)
# assert not (SAVE_HEATMAP_NPY and DECODE_HEATMAP_TO_CSV) and (SAVE_HEATMAP_NPY or DECODE_HEATMAP_TO_CSV)


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


if DECODE_HEATMAP_TO_CSV:
    all_agg_dfs = {}
    for agg_name in ALL_UNIQUE_AGG_NAMES:
        logger.info("PROCESSING AGG_NAME=%s", agg_name)
        chunk_agg_dfs = []
        for chunk_idx in range(len(DEVICES)):
            csv_name = f"{agg_name}_chunk{chunk_idx}.csv"
            csv_path = os.path.join(TMP_CSV_DIR, csv_name)
            try:
                df = pd.read_csv(csv_path)
                chunk_agg_dfs.append(df)
                logger.info(
                    "Append new chunk Dataframe with shape %s, columns=%s",
                    df.shape,
                    list(df.columns),
                )
            except Exception as e:
                logger.warning(
                    "EXCEPTION OCCUR when processing agg_name=%s:\n%s\nIGNORE CSV RESULT FILE AT %s",
                    agg_name,
                    e,
                    csv_path,
                )
        agg_df = pd.concat(chunk_agg_dfs, axis=0, ignore_index=True).reset_index(
            drop=True
        )
        logger.info(
            "CONCAT ALL TEMPORARY DATAFRAMES INTO A SINGLE DATAFRAME WITH SHAPE %s, COLUMNS=%s\n%s",
            df.shape,
            list(df.columns),
            df,
        )
        all_agg_dfs[agg_name] = agg_df
        print("-------------------------------------------\n\n")
    logger.info("TOTAL %d AGG DataFrame", len(all_agg_dfs))

    if CSV_ENSEMBLE_MODE == "max":
        df = pd.concat(
            list(all_agg_dfs.values()), axis=0, ignore_index=True
        ).reset_index(drop=True)
        df = (
            df.sort_values("conf", ascending=False)
            .groupby("tomo_id", as_index=False)
            .first()
        )
    elif CSV_ENSEMBLE_MODE == "wbf":
        submission_df = SubmissionDataFrame()
        assert len(ALL_TOMO_IDS) == len(ALL_TOMO_SPACINGS)
        for tomo_idx, tomo_id in tqdm(enumerate(ALL_TOMO_IDS), desc="WBF"):
            tomo_zyxs = []
            tomo_confs = []
            tomo_labels = []
            weights = []
            for agg_name, agg_df in all_agg_dfs.items():
                preds = agg_df[agg_df["tomo_id"] == tomo_id][
                    ["motor_z", "motor_y", "motor_x", "conf"]
                ].to_numpy()
                keep_idxs = np.all(preds[:, :3] != -1, axis=1) & (preds[:, -1] > 0.0)
                preds = preds[keep_idxs]
                tomo_zyxs.append(preds[:, :3])
                tomo_confs.append(preds[:, 3])
                tomo_labels.append([0] * len(preds))
                weights.append(ALL_UNIQUE_AGG_WEIGHTS[agg_name])
            # print(len(tomo_zyxs), len(tomo_confs), len(tomo_labels))
            # print(tomo_zyxs, tomo_confs, tomo_labels, sep = '\n\n')
            zyxs, confs, labels = weighted_boxes_fusion_3d(
                tomo_zyxs,
                tomo_confs,
                tomo_labels,
                weights=weights,
                dist_thr=1000.0 / ALL_TOMO_SPACINGS[tomo_idx],
                skip_box_thr=WBF_CONF_THRES,
                conf_type=WBF_CONF_TYPE,
                allows_overflow=False,
                rescale_mode="poly",  # None | linear | clipping | poly
                rescale_clipping=3,
                rescale_poly=1,
            )
            if len(zyxs):
                # select the highest conf
                select_idx = np.argmax(confs)
                z, y, x = zyxs[select_idx]
                submission_df.add_row(tomo_id, x, y, z, confs[select_idx])
            else:
                submission_df.add_row(tomo_id, -1, -1, -1, 0.0)
        df = submission_df.to_pandas(submit=False)
    else:
        raise ValueError

logger.info(
    "PREDICTION DATAFRAME WITH SHAPE %s COLUMNS=%s:\n%s",
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
    # assert 0 <= filter_thres <= 1.0


logger.info("USING FILTER THRESHOLD: %f", filter_thres)
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
