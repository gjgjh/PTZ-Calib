'''
Author: mingjian (lory.gjh@alibaba-inc.com)
Created Date: 2025-02-18 16:53:40
Modified By: mingjian (lory.gjh@alibaba-inc.com)
Last Modified: 2025-03-24 20:04:59
-----
Copyright (c) 2025 Alibaba Inc.
'''

import json
import cv2
import numpy as np
import argparse
from typing import List, Tuple
from shapely.geometry import Polygon

# gt model cooridinates (yard), data from BBInference/EvaluateBaseLine.cpp
FIELD_X = 114.83
FIELD_Y = 74.37
# gt model cooridinates (meters)
# FIELD_X = 105.000552
# FIELD_Y = 68.003928


def read_json(filepath: str):
    with open(filepath, encoding='utf-8', mode='r') as f:
        j = json.load(f)
        return j


def read_H(H_path):
    with open(H_path, "r") as f:
        lines = f.readlines()

    H = []
    for i in range(3):
        line = lines[i]
        words = line.strip().split()
        H.append([float(words[0]),  float(words[1]), float(words[2])])

    return np.array(H)


def H_from_KRt(K, R, t):
    """
    从KRt计算H
    """
    RT = np.column_stack((R[:, :2], t))
    RT[:, 1] *= -1  # 我们的坐标系定义中Z轴朝上, Y轴正向和原数据集的定义相反
    pred_H_inv = K @ RT

    assert (np.linalg.det(pred_H_inv) !=
            0 and "pred_H_inv = K @ RT is singular, cannot compute H")

    pred_H = np.linalg.inv(pred_H_inv)
    pred_H /= pred_H[2, 2]

    return pred_H


def calc_iou_part_KRt(K, R, t, dist, gt_H):
    """
    将图像投影到3D BEV空间(在[template_w, template_h]范围内得到的仅仅是field), 再基于field mask进行iou计算
    ref: https://github.com/ericsujw/KpSFR/blob/main/metrics.py
    """
    template_w = round(FIELD_X)
    template_h = round(FIELD_Y)
    frame_w = 1280
    frame_h = 720

    field_mask = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 255

    field_mask_undist = cv2.undistort(field_mask, K, dist, K)
    pred_H = H_from_KRt(K, R, t)

    gt_mask = cv2.warpPerspective(field_mask, gt_H, (template_w, template_h),
                                  cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    pred_mask = cv2.warpPerspective(field_mask_undist, pred_H, (template_w, template_h),
                                    cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    # cv2.imwrite("gt_mask.jpg", gt_mask)
    # cv2.imwrite("pred_mask.jpg", pred_mask)
    gt_mask[gt_mask > 0] = 255
    pred_mask[pred_mask > 0] = 255

    intersection = ((gt_mask > 0) * (pred_mask > 0)).sum()
    union = (gt_mask > 0).sum() + (pred_mask > 0).sum() - intersection

    if union <= 0:
        print('part union', union)
        # iou = float('nan')
        iou = 0.
    else:
        iou = float(intersection) / float(union)

    return iou


def calc_iou_whole_KRt(K, R, t, dist, gt_H):
    """
    KRt版本
    ref: https://github.com/ericsujw/KpSFR/blob/main/metrics.py
    """
    frame_w = 1280
    frame_h = 720

    corners = np.array([[0, 0],
                        [frame_w - 1, 0],
                        [frame_w - 1, frame_h - 1],
                        [0, frame_h - 1]], dtype=np.float64)
    gt_corners_3d = cv2.perspectiveTransform(
        corners[:, None, :], gt_H)  # inv_gt_mat * (gt_mat * [x, y, 1])
    gt_corners_2d = cv2.perspectiveTransform(
        gt_corners_3d, np.linalg.inv(gt_H))
    gt_corners_2d = gt_corners_2d[:, 0, :]

    gt_corners_3d = gt_corners_3d[:, 0, :]
    gt_corners_3d = np.column_stack((gt_corners_3d, np.array([0, 0, 0, 0])))
    gt_corners_3d[:, 1] *= -1   # 我们的坐标系定义中Z轴朝上, Y轴正向和原数据集的定义相反
    rvec, _ = cv2.Rodrigues(R)
    pred_corners_2d, _ = cv2.projectPoints(gt_corners_3d, rvec, t, K, dist)
    pred_corners_2d = pred_corners_2d[:, 0, :]

    # For debug
    # img = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 255
    # cv2.fillPoly(img, [pred_corners_2d.astype(np.int64)], [0,  255, 0])
    # cv2.polylines(img, [gt_corners_2d.astype(np.int64)],
    #               True, [0,  0, 255], thickness=2)
    # cv2.imwrite("pred_corners.jpg", img)

    gt_poly = Polygon(gt_corners_2d.tolist())
    pred_poly = Polygon(pred_corners_2d.tolist())

    if pred_poly.is_valid is False:
        return 0.

    if not gt_poly.intersects(pred_poly):
        print('not intersects')
        iou = 0.
    else:
        intersection = gt_poly.intersection(pred_poly).area
        union = gt_poly.area + pred_poly.area - intersection
        if union <= 0.:
            print('whole union', union)
            iou = 0.
        else:
            iou = intersection / union

    return iou


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate results on WorldCup14 dataset')
    parser.add_argument('--pred', type=str, required=True,
                        help='Path to prediction file')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Ground-truth directory (containing .homographyMatrix files)')
    args = parser.parse_args()

    return args


def eval_worldcup(pred_path: str, gt_dir: str) -> Tuple[List[float], List[float]]:
    iou_part_list = []
    iou_whole_list = []

    pred = read_json(pred_path)
    for test_rootname, params in pred["cameras"].items():
        index = test_rootname.split("-")[1]

        gt_H_path = f"{gt_dir}/{index}.homographyMatrix"
        gt_H = read_H(gt_H_path)

        K = np.array(params["K"]).reshape(3, 3)
        R = np.array(params["R"]).reshape(3, 3)
        t = np.array(params["t"]).reshape(3, 1)
        dist = np.array(params["dist"]).reshape(5, 1).astype('float')

        iou_part = calc_iou_part_KRt(K, R, t, dist, gt_H)
        iou_part_list.append(iou_part)
        iou_whole = calc_iou_whole_KRt(K, R, t, dist, gt_H)
        iou_whole_list.append(iou_whole)

    mean_iou_part = np.nanmean(iou_part_list)
    mean_iou_whole = np.nanmean(iou_whole_list)
    median_iou_part = np.nanmedian(iou_part_list)
    median_iou_whole = np.nanmedian(iou_whole_list)

    print(
        f"Mean IOU whole: {mean_iou_whole * 100.:.1f}, Median IOU whole: {median_iou_whole * 100.:.1f}")
    print(
        f"Mean IOU part: {mean_iou_part * 100.:.1f}, Median IOU part: {median_iou_part * 100.:.1f}")

    return iou_part_list, iou_whole_list


def eval_worldcup_all(pred_paths: List[str], gt_dir: str) -> None:
    iou_part_list_all = []
    iou_whole_list_all = []

    for pred_path in pred_paths:
        iou_part_list, iou_whole_list = eval_worldcup(pred_path, gt_dir)
        iou_part_list_all += iou_part_list
        iou_whole_list_all += iou_whole_list

    mean_iou_part = np.nanmean(iou_part_list_all)
    mean_iou_whole = np.nanmean(iou_whole_list_all)
    median_iou_part = np.nanmedian(iou_part_list_all)
    median_iou_whole = np.nanmedian(iou_whole_list_all)

    print(f"All average")
    print(
        f"Mean IOU whole: {mean_iou_whole * 100.:.1f}, Median IOU whole: {median_iou_whole * 100.:.1f}")
    print(
        f"Mean IOU part: {mean_iou_part * 100.:.1f}, Median IOU part: {median_iou_part * 100.:.1f}")


if __name__ == "__main__":
    args = parse_args()
    pred_path = args.pred
    gt_dir = args.gt_dir

    print(f"Start evaluating {pred_path} ...")
    eval_worldcup(pred_path, gt_dir)
