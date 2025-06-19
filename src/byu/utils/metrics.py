import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics
from scipy.spatial import KDTree
from yagm.metrics.binary import compute_fbeta_score
from yagm.metrics.detection import (
    compute_partial_average_precision_score,
    compute_precision_recall_curve_fast,
)

logger = logging.getLogger(__name__)


COORD_COLS = ["motor_z", "motor_y", "motor_x"]
RADIUS = 1000
METRIC_THRESHOLD_RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
METRIC_MAIN_THRESHOLD_RATIO = 1.0
PAP_RECALL_CUTOFF = 0.4


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def kaggle_distance_metric(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    thresh_ratio: float,
    min_radius: float,
):
    label_tensor = solution[COORD_COLS].values.reshape(
        len(solution), -1, len(COORD_COLS)
    )  # (N, 1, 3)
    predicted_tensor = submission[COORD_COLS].values.reshape(
        len(submission), -1, len(COORD_COLS)
    )  # (N, 1, 3)
    # Find the minimum euclidean distances between the true and predicted points
    solution["distance"] = np.linalg.norm(label_tensor - predicted_tensor, axis=2).min(
        axis=1
    )
    # Convert thresholds from angstroms to voxels
    solution["thresholds"] = solution["voxel_spacing"].apply(
        lambda x: (min_radius * thresh_ratio) / x
    )
    solution["predictions"] = submission["has_motors"].values
    solution.loc[
        (solution["distance"] > solution["thresholds"])
        & (solution["has_motors"] == 1)
        & (submission["has_motors"] == 1),
        "predictions",
    ] = 0
    return solution["predictions"].values


def kaggle_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    min_radius: float = RADIUS,
    beta: float = 2,
) -> float:
    """
    Parameters:
    solution (pd.DataFrame): DataFrame containing ground truth motor positions.
    submission (pd.DataFrame): DataFrame containing predicted motor positions.

    Returns:
    float: FBeta score.

    Example
    --------
    >>> solution = pd.DataFrame({
    ...     'tomo_id': [0, 1, 2, 3],
    ...     'motor_z': [-1, 250, 100, 200],
    ...     'motor_y': [-1, 250, 100, 200],
    ...     'motor_x': [-1, 250, 100, 200],
    ...     'voxel_spacing': [10, 10, 10, 10],
    ...     'has_motors': [0, 1, 1, 1]
    ... })
    >>> submission = pd.DataFrame({
    ...     'tomo_id': [0, 1, 2, 3],
    ...     'motor_z': [100, 251, 600, -1],
    ...     'motor_y': [100, 251, 600, -1],
    ...     'motor_x': [100, 251, 600, -1]
    ... })
    >>> score(solution, submission, 1000, 2)
    0.3571428571428571
    """
    # do copy already
    solution = solution.sort_values("tomo_id").reset_index(drop=True)
    solution["has_motors"] = solution["num_motors"] > 0
    submission = submission.sort_values("tomo_id").reset_index(drop=True)

    filename_equiv_array = (
        solution["tomo_id"].eq(submission["tomo_id"], fill_value=0).values
    )

    if np.sum(filename_equiv_array) != len(solution["tomo_id"]):
        raise ValueError(
            "Submitted tomo_id values do not match the sample_submission file"
        )

    submission["has_motors"] = 1
    # If any columns are missing an axis, it's marked with no motor
    select = (submission[COORD_COLS] == -1).any(axis="columns")
    submission.loc[select, "has_motors"] = 0

    cols = ["has_motors", *COORD_COLS]
    assert all(col in submission.columns for col in cols)

    # Calculate a label of 0 or 1 using the 'has_motors', and 'motor axis' values
    predictions = kaggle_distance_metric(
        solution,
        submission,
        thresh_ratio=1.0,
        min_radius=min_radius,
    )
    fbeta = sklearn.metrics.fbeta_score(
        solution["has_motors"].values, predictions, beta=beta
    )
    precision = sklearn.metrics.precision_score(
        solution["has_motors"].values, predictions
    )
    recall = sklearn.metrics.recall_score(solution["has_motors"].values, predictions)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        solution["has_motors"].values, predictions
    ).ravel()
    return fbeta, precision, recall, (tn, fp, fn, tp)


def match_over_thresholds(
    gt_coords,
    gt_has_motors,
    pred_coords,
    pred_confs,
    voxel_radiuses,
    thres_ratios: List[float],
):
    """
    Args:
        gt_coords: (N, 3)
        gt_has_motors: (N,)
        pred_coords: (N, 3)
        voxel_radiuses: radius threshold (currently 1000A) in voxels space
        thres_ratios: multiplication factor to thresholds
    """
    # Find the minimum euclidean distances between the true and predicted points
    dists = np.linalg.norm(gt_coords - pred_coords, axis=1)
    pred_has_motors = (pred_confs > 0) & (pred_coords != -1).all(axis=1)
    ret = {}
    for thres_ratio in thres_ratios:
        # matches (np.ndarray): A binary array where 1 indicates a true positive (TP) and 0 indicates a false positive (FP).
        matches = gt_has_motors & (dists <= voxel_radiuses * thres_ratio)
        ret[thres_ratio] = (matches[pred_has_motors], pred_confs[pred_has_motors])
    return ret


def compute_metrics(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    thres_ratios: Tuple[float] = METRIC_THRESHOLD_RATIOS,
    mode="kaggle",
) -> float:
    """
    Args"
        solution: GT dataframe
        submission: PREDICTION dataframe
        thres_ratios: multiplication factor to the base radius thresholds
        mode: `kaggle` or `detection`. For one tomo with 1 GT motor,
            if prediction ZYX coordinate is outside of GT
            `kaggle`: as implemented in official Kaggle metric, treat problem as binary classification and just +1 FN
            `detection`: +1 FN and +1 FP, result in same Recall but lower Precision compared to `kaggle` mode
    Returns:
        dictionary of computed metrics
    """
    solution = solution.sort_values("tomo_id").reset_index(drop=True)
    submission = submission.sort_values("tomo_id").reset_index(drop=True)

    assert solution["tomo_id"].eq(submission["tomo_id"]).all()

    gt_coords = solution[COORD_COLS].values  # (N, 3)
    pred_coords = submission[COORD_COLS].values  # (N, 3)
    pred_confs = submission["conf"].values  # (N, )
    gt_has_motors = (solution["num_motors"] > 0).values  # (N, )
    assert ((gt_coords == -1).any(axis=1) != gt_has_motors).all()
    voxel_radiuses = (
        RADIUS / solution["voxel_spacing"]
    ).values  # threshold in voxels space

    if mode == "detection":
        matches_over_thresholds = match_over_thresholds(
            gt_coords,
            gt_has_motors,
            pred_coords,
            pred_confs,
            voxel_radiuses,
            thres_ratios,
        )
        num_gt_pos = gt_has_motors.sum()
    elif mode == "kaggle":
        dists = np.linalg.norm(gt_coords - pred_coords, axis=1)
        pred_has_motors = (pred_confs > 0) & (pred_coords != -1).all(axis=1)
    else:
        raise ValueError

    def _add(d, key, value):
        assert key not in d
        d[key] = value

    metrics = {}
    all_aps = []
    all_paps = []
    for thres_ratio in thres_ratios:
        if mode == "detection":
            (matches, confs) = matches_over_thresholds[thres_ratio]
            precisions, recalls, conf_thresholds = compute_precision_recall_curve_fast(
                matches, confs, num_gt_pos
            )
        elif mode == "kaggle":
            preds = pred_confs.copy()
            preds[
                gt_has_motors & pred_has_motors & (dists > voxel_radiuses * thres_ratio)
            ] = -1.0
            precisions, recalls, conf_thresholds = (
                sklearn.metrics.precision_recall_curve(gt_has_motors, preds)
            )
            if conf_thresholds[0] == -1:
                precisions, recalls, conf_thresholds = (
                    precisions[1:],
                    recalls[1:],
                    conf_thresholds[1:],
                )
        else:
            raise ValueError

        assert len(precisions) == len(recalls) == len(conf_thresholds) + 1
        assert precisions[-1] == 1.0 and recalls[-1] == 0.0

        ap = compute_partial_average_precision_score(
            precisions, recalls, recall_cutoff=0.0, rescale=False
        )
        pap = compute_partial_average_precision_score(
            precisions,
            recalls,
            recall_cutoff=PAP_RECALL_CUTOFF,
            rescale=True,
        )
        all_aps.append(ap)
        all_paps.append(pap)
        if thres_ratio == METRIC_MAIN_THRESHOLD_RATIO:
            best_recall = max(recalls[:-1]) if len(recalls) > 1 else recalls[0]
            fbetas = compute_fbeta_score(precisions, recalls, beta=2)
            best_fbeta_idx = np.argmax(fbetas)
            best_fbeta = fbetas[best_fbeta_idx]
            best_fbeta_precision = precisions[best_fbeta_idx]
            best_fbeta_recall = recalls[best_fbeta_idx]
            best_fbeta_conf_thres = (
                conf_thresholds[best_fbeta_idx] if len(conf_thresholds) else 0.5
            )
            _add(metrics, "Fbeta", best_fbeta)
            _add(metrics, "Precision", best_fbeta_precision)
            _add(metrics, "Recall", best_fbeta_recall)
            _add(metrics, "thres", best_fbeta_conf_thres)
            _add(metrics, "AP", ap)
            _add(metrics, "PAP", pap)
            _add(metrics, "bestR", best_recall)

    map = np.mean(all_aps)
    mpap = np.mean(all_paps)
    _add(metrics, "mAP", map)
    _add(metrics, "mPAP", mpap)

    best_thres = metrics["thres"]
    submission.loc[submission["conf"] < best_thres, COORD_COLS] = -1
    kaggle_fbeta, kaggle_precision, kaggle_recall, _ = kaggle_score(
        solution, submission, min_radius=RADIUS, beta=2
    )
    _add(metrics, "kaggleFbeta", kaggle_fbeta)
    return metrics


if __name__ == "__main__":
    # gt_df = pd.read_csv("/home/dangnh36/datasets/byu/processed/gt.csv")
    # print(list(gt_df.columns))
    # print(gt_df)
    # pred_df = gt_df.copy()
    # pred_df["conf"] = 0.9
    # # pred_df.loc[pred_df["motor_x"] == -1, "conf"] = 0.0
    # pred_df.loc[pred_df["motor_x"] == -1, "conf"] = np.random.random(
    #     (pred_df["num_motors"] == 0).sum()
    # )
    # pred_df.loc[pred_df["motor_x"] == -1, COORD_COLS] = 12.345
    # metrics = compute_metrics(gt_df, pred_df)
    # print("METRICS:\n", metrics)

    # thres = 0.3
    # thres = metrics["thres"]
    # pred_df.loc[pred_df["conf"] < thres, COORD_COLS] = -1
    # print("Kaggle metric:", kaggle_score(gt_df, pred_df, min_radius=RADIUS, beta=2))

    import json

    PRED_PATH = "/home/dangnh36/projects/.comp/byu/outputs/run/multirun/04-09/02-53-30.100047_unet_x3d-spacing64/0_cv.fold_idx=0,data.fast_val_workers=8,data.io_backend=cv2,data.patch_size=[256,320,320],data.transform.heatmap_stride=4,data.transform.resample_mode=bilinear,data.transform.target_spacing=64.0,exp=unet_x3d,exp_name=unet_x3d-spacing64,loader.train_b_ETC_/fold_0/metadata/pred/ema_0.99/ep=8_step=10000.csv"
    gt_df = pd.read_csv("/home/dangnh36/datasets/.comp/byu/processed/gt.csv")
    with open(
        "/home/dangnh36/datasets/.comp/byu/processed/cv/skf5_rd42.json", "r"
    ) as f:
        cv_meta = json.load(f)
    val_tomo_ids = cv_meta["folds"][0]["val"]
    gt_df = gt_df[gt_df["tomo_id"].isin(val_tomo_ids)].reset_index(drop=True)
    print(list(gt_df.columns), gt_df.shape)
    print(gt_df)

    pred_df = pd.read_csv(PRED_PATH)
    pred_df = pred_df[
        pred_df["tomo_id"].apply(lambda x: x.split("@")[1]) == "zyx"
    ].reset_index(drop=True)
    pred_df["tomo_id"] = pred_df["tomo_id"].apply(lambda x: x.split("@")[0])
    print(pred_df.columns, pred_df.shape)
    print(pred_df)

    metrics = compute_metrics(gt_df, pred_df, mode="kaggle")
    print("METRICS:\n", metrics)
    # thres = 0.3
    thres = metrics["thres"]
    print(f"USING THRESHOLD={thres}")
    print(f"NUMBER OF POSITIVE:", (pred_df["conf"] >= thres).sum())
    pred_df.loc[pred_df["conf"] < thres, COORD_COLS] = -1
    kaggle_fbeta, kaggle_precision, kaggle_recall, _ = kaggle_score(
        gt_df, pred_df, min_radius=RADIUS, beta=2
    )
    print("Kaggle Fbeta:", kaggle_fbeta)
    print("Kaggle Precision:", kaggle_precision)
    print("Kaggle Recall:", kaggle_recall)
