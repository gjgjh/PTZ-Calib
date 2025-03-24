/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 15:59:48
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-03-12 16:43:25
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#include <unordered_set>

#include "core/types.h"
#include "utils/cmdline.h"

cmdline::parser ParseArgs(int argc, char** argv);

bool RunPtzBA(const std::vector<std::string>& fnames, const std::vector<ptzcalib::ImageFeatures>& features,
              const std::vector<ptzcalib::MatchesInfo>& matches_info, int max_iter, std::vector<ptzcalib::Camera>& cameras,
              std::unordered_set<long>& reg_image_ids);
bool RunGeoreferencing(const std::vector<ptzcalib::ImageFeatures>& features, const std::vector<ptzcalib::MatchesInfo>& matches_info,
                       const std::vector<std::vector<cv::Point2f>>& pixels, const std::vector<std::vector<cv::Point3d>>& pts3d,
                       const std::unordered_set<long>& cam_ids, int max_iter, bool has_dist, std::vector<ptzcalib::Camera>& cameras,
                       double& error_2d2d, double& error_2d3d);