/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-03-11 14:50:07
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-03-24 15:23:14
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#ifndef PTZ_CALIB_SRC_APP_RUN_PTZ_RELOC_H
#define PTZ_CALIB_SRC_APP_RUN_PTZ_RELOC_H

#include "core/types.h"
#include "utils/cmdline.h"

typedef std::pair<std::string, std::vector<cv::DMatch>> BestMatchT;

cmdline::parser ParseArgs(int argc, char** argv);

BestMatchT FindBestMatch(const std::string& fname, const std::vector<std::pair<std::string, std::string>>& img_pairs_name,
                         const std::vector<std::vector<cv::DMatch>>& pairs_matches);

cv::Mat VisMatching(const cv::Mat &img1, const std::vector<cv::KeyPoint> &kpts1, const cv::Mat &img2,
                                      const std::vector<cv::KeyPoint> &kpts2, const std::vector<cv::DMatch> &matches12);

#endif  // PTZ_CALIB_SRC_APP_RUN_PTZ_RELOC_H
