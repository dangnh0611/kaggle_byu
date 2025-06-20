"""
Modified version of https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf_3d.py (MIT License)
to ensemble 3D points localization from multiple models. Using L2 distance for Clustering
"""

# coding: utf-8
__author__ = "ZFTurbo: https://kaggle.com/zfturbo"


import warnings

import numpy as np


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print(
                "Error. Length of boxes arrays not equal to length of scores array: {} != {}".format(
                    len(boxes[t]), len(scores[t])
                )
            )
            exit()

        if len(boxes[t]) != len(labels[t]):
            print(
                "Error. Length of boxes arrays not equal to length of labels array: {} != {}".format(
                    len(boxes[t]), len(labels[t])
                )
            )
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            z = float(box_part[0])
            y = float(box_part[1])
            x = float(box_part[2])
            b = [int(label), float(score) * weights[t], z, y, x]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type="avg"):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """

    box = np.zeros(5, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        box[2:] += b[1] * b[2:]
        conf += b[1]
        conf_list.append(b[1])
    box[0] = boxes[0][0]
    if conf_type == "avg":
        box[1] = conf / len(boxes)
    elif conf_type == "max":
        box[1] = np.array(conf_list).max()
    box[2:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_dist):

    def _l2_dist(kpt1, kpt2):
        return ((kpt1 - kpt2) ** 2).sum() ** 0.5

    best_dist = match_dist
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        dist = _l2_dist(box[2:], new_box[2:])
        if dist < best_dist:
            best_index = i
            best_dist = dist
    return best_index, best_dist


def weighted_boxes_fusion_3d(
    boxes_list,
    scores_list,
    labels_list,
    weights=None,
    dist_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    allows_overflow=False,
    rescale_mode="linear",  # linear | clipping | poly
    rescale_clipping=3,
    rescale_poly=1,
):
    """
    :param boxes_list: list of boxes predictions from each model, each box is 6 numbers.
    It has 3 dimensions (models_number, model_preds, 6)
    Order of boxes: x1, y1, z1, x2, y2 z2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, z1, x2, y2, z2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    """

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print(
            "Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.".format(
                len(weights), len(boxes_list)
            )
        )
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ["avg", "max"]:
        print(
            'Error. Unknown conf_type: {}. Must be "avg" or "max". Use "avg"'.format(
                conf_type
            )
        )
        conf_type = "avg"

    filtered_boxes = prefilter_boxes(
        boxes_list, scores_list, labels_list, weights, skip_box_thr
    )
    if len(filtered_boxes) == 0:
        return np.zeros((0, 3)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_dist = find_matching_box(weighted_boxes, boxes[j], dist_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())

        # Rescale confidence based on number of models and boxes
        if rescale_mode is not None:
            for i in range(len(new_boxes)):
                # by default, scale by number of element inside this cluster
                rescale_factor = len(new_boxes[i])
                if rescale_mode == "linear":
                    pass
                elif rescale_mode == "clipping":
                    rescale_factor = min(rescale_factor, rescale_clipping)
                elif rescale_mode == "poly":
                    rescale_factor = rescale_factor**rescale_poly
                else:
                    raise ValueError
                if not allows_overflow:
                    weighted_boxes[i][1] = (
                        weighted_boxes[i][1]
                        * min(weights.sum(), rescale_factor)
                        / weights.sum()
                    )
                else:
                    weighted_boxes[i][1] = (
                        weighted_boxes[i][1] * rescale_factor / weights.sum()
                    )

        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels
