/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 17:26:56
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-03-12 16:46:47
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#ifndef PTZ_CALIB_SRC_CORE_DATA_IO_H
#define PTZ_CALIB_SRC_CORE_DATA_IO_H

#include <string>
#include <unordered_set>

#include "core/types.h"

namespace ptzcalib {

/** Read Colmap format features from file.
 * Colmap features format: https://colmap.github.io/tutorial.html#feature-detection-and-extraction
 */
void ReadColmapFeatures(const std::string &filepath, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc);

/** Read Colmap format matches from file.
 * Colmap matches format: https://colmap.github.io/tutorial.html#feature-matching-and-geometric-verification
 */
void ReadColmapMatches(const std::string &filepath, std::vector<std::vector<cv::DMatch>> &pairs_matches,
                       std::vector<std::pair<std::string, std::string>> &img_pairs_name);

bool SaveToJson(const std::vector<Camera> &cameras, const std::vector<std::string> &names,
                const std::vector<std::vector<cv::Point2f>> &pixels_gt, const std::vector<std::vector<cv::Point3d>> &pts3d_gt,
                const std::string &filepath);

bool ReadFromJson(const std::string &filepath, std::vector<Camera> &cameras, std::vector<std::string> &names,
                  std::vector<std::vector<cv::Point2f>> &pixels, std::vector<std::vector<cv::Point3d>> &pts3d,
                  std::vector<cv::Size> &sizes);

bool ReadCamFromJson(const std::string &filepath, const std::vector<std::string> &names, std::vector<Camera> &cameras);

bool LoadImgsAndFeatures(const std::string& img_dir, const std::string& feature_dir, std::vector<std::string>& fnames,
                         std::vector<ptzcalib::ImageFeatures>& features, std::vector<cv::Size>& sizes);

bool LoadMatchesInfo(const std::string& matches_path, const std::vector<std::string>& fnames,
                     const std::vector<ptzcalib::ImageFeatures>& features, std::vector<ptzcalib::MatchesInfo>& matches_info);
                     
bool LoadAnnotation(const std::string& annot_path, const std::vector<std::string>& fnames, std::vector<std::vector<cv::Point2f>>& pixels,
                    std::vector<std::vector<cv::Point3d>>& pts3d);

void SaveRegisteredCam(const std::vector<ptzcalib::Camera>& cameras, const std::unordered_set<long>& reg_image_ids,
                       const std::vector<std::string>& fnames, const std::vector<std::vector<cv::Point2f>>& pixels,
                       const std::vector<std::vector<cv::Point3d>>& pts3d, const std::string& out_dir);

long FindImgIndex(const std::vector<std::string>& fnames, const std::string& fname);

}  // namespace ptzcalib

#endif  // PTZ_CALIB_SRC_CORE_DATA_IO_H