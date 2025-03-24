'''
Author: mingjian (lory.gjh@alibaba-inc.com)
Created Date: 2025-02-18 16:53:11
Modified By: mingjian (lory.gjh@alibaba-inc.com)
Last Modified: 2025-02-18 17:52:40
-----
Copyright (c) 2025 Alibaba Inc.
'''


import json
import math
import argparse
import numpy as np
from scipy.spatial.transform import Rotation


def read_json(filepath: str):
    with open(filepath, encoding='utf-8', mode='r') as f:
        j = json.load(f)
        return j


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate results on synthetic dataset')
    parser.add_argument('--pred', type=str, required=True,
                        help='Path to prediction file')
    parser.add_argument('--gt', type=str, required=True,
                        help='Path to ground-truth file')
    args = parser.parse_args()

    return args


def calc_focal_error(pred_f, gt_f):
    diff_f = abs(pred_f - gt_f)
    return diff_f


def calc_ape(pred_R, pred_t, gt_R, gt_t):
    """
    Calculate absolute pose error (APE)
    ref: https://github.com/MichaelGrupp/evo/blob/master/notebooks/metrics.py_API_Documentation.ipynb
    """
    pred_P = np.column_stack((pred_R, pred_t))
    pred_P = np.row_stack((pred_P, np.array([0, 0, 0, 1])))
    gt_P = np.column_stack((gt_R, gt_t))
    gt_P = np.row_stack((gt_P, np.array([0, 0, 0, 1])))

    relative_P = pred_P @ np.linalg.inv(gt_P)
    relative_P /= relative_P[3, 3]

    relative_R = relative_P[0:3, 0:3]
    relative_t = relative_P[0:3, 3]

    tvec = -relative_R.T @ relative_t
    ape_trans = np.linalg.norm(tvec)

    r = Rotation.from_matrix(relative_R)
    rvec = r.as_rotvec()
    ape_rot = np.linalg.norm(rvec)
    ape_rot = math.degrees(ape_rot)

    return ape_trans, ape_rot


def cal_mean_median(data_list):
    data_array = np.array(data_list)

    mean = np.nanmean(data_array)
    median = np.nanmedian(data_array)

    return mean, median


def eval_synthetic():
    args = parse_args()

    j_pred = read_json(args.pred)
    j_gt = read_json(args.gt)

    focal_error_abs_list = []
    ape_rot_error_list = []
    ape_trans_error_list = []

    keys = list(j_pred["cameras"].keys())
    for key in keys:
        K = np.array(j_pred["cameras"][key]["K"]).reshape(3, 3)
        f = K[0, 0]
        R = np.array(j_pred["cameras"][key]["R"]).reshape(3, 3)
        t = np.array(j_pred["cameras"][key]["t"]).reshape(3, 1)

        K_gt = np.array(j_gt["cameras"][key]["K"]).reshape(3, 3)
        f_gt = K_gt[0, 0]
        R_gt = np.array(j_gt["cameras"][key]["R"]).reshape(3, 3)
        t_gt = np.array(j_gt["cameras"][key]["t"]).reshape(3, 1)

        focal_error_abs = calc_focal_error(f, f_gt)
        focal_error_abs_list.append(focal_error_abs)

        ape_trans, ape_rot = calc_ape(R, t, R_gt, t_gt)
        ape_rot_error_list.append(ape_rot)
        ape_trans_error_list.append(ape_trans)

    print(f"Total sample number: {len(keys)}")
    print(f"focal_error_abs [mean, median]: {cal_mean_median(focal_error_abs_list)[0] :.2f}, {cal_mean_median(focal_error_abs_list)[1] :.2f}"
          )
    print(
        f"ape_rot [mean, median]: {cal_mean_median(ape_rot_error_list)[0]: .2f}, {cal_mean_median(ape_rot_error_list)[1]: .2f}")
    print(
        f"ape_trans [mean, median]: {cal_mean_median(ape_trans_error_list)[0]: .2f}, {cal_mean_median(ape_trans_error_list)[1]: .2f}")


if __name__ == "__main__":
    eval_synthetic()
