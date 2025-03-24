/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 15:34:47
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-03-12 16:43:12
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#include "run_ptz_ba.h"

#include <opencv2/highgui.hpp>
#include <unordered_set>

#include "core/data_io.h"
#include "core/ptz_incremental_optimizer.h"
#include "core/ptzray_optimizer.h"
#include "utils/logging.h"
#include "utils/os_path.h"

using namespace std;
using namespace ptzcalib;

int main(int argc, char** argv)
{
  InitLogging(false);

  cmdline::parser parser = ParseArgs(argc, argv);

  vector<string> fnames;
  vector<ImageFeatures> features;
  vector<cv::Size> sizes;
  bool load_status = LoadImgsAndFeatures(parser.get<string>("images"), parser.get<string>("features"), fnames, features, sizes);
  if (!load_status) {
    PLOGE << "Error loading images and features. Exiting ...";
    return -1;
  }

  vector<MatchesInfo> matches_info;
  string matches_path = parser.get<string>("features") + "/pairs_matches.txt";
  load_status = LoadMatchesInfo(matches_path, fnames, features, matches_info);
  if (!load_status) {
    PLOGE << "Error loading matches from " << matches_path << ". Exiting ...";
    return -1;
  }
  else {
    PLOGI << "================== PTZ-IBA Begin ==========================";
  }

  vector<Camera> cameras;
  unordered_set<long> reg_image_ids;
  static const int MAX_ITER = 200;
  bool ba_status = RunPtzBA(fnames, features, matches_info, MAX_ITER, cameras, reg_image_ids);
  if (!ba_status) {
    PLOGI << "================== PTZ-IBA End: failed ==========================";
    return -1;
  }
  else {
    PLOGI << "================== PTZ-IBA End: success ==========================";
  }

  vector<vector<cv::Point2f>> pixels;
  vector<vector<cv::Point3d>> pts3d;
  load_status = LoadAnnotation(parser.get<string>("annotation"), fnames, pixels, pts3d);
  if (!load_status) {
    PLOGE << "Error loading annotation from " << parser.get<string>("annotation") << ". Exiting ...";
    return -1;
  }
  else {
    PLOGI << "Success loading annotation from " << parser.get<string>("annotation");
    PLOGI << "================== Georeferencing Begin ==========================";
  }

  double error_2d2d, error_2d3d;
  bool georef_status = RunGeoreferencing(features, matches_info, pixels, pts3d, reg_image_ids, MAX_ITER, parser.exist("dist"), cameras,
                                         error_2d2d, error_2d3d);
  if (!georef_status) {
    PLOGI << "================== Georeferencing End: failed ==========================";
    return -1;
  }
  else {
    PLOGI << "================== Georeferencing End: success ==========================";
  }

  string cam_id = basename(parser.get<string>("images"));
  string out_dir = parser.get<string>("output");
  mkdir_ifnot_exist(out_dir);
  string out_path = out_dir + "/" + cam_id + ".json";

  SaveRegisteredCam(cameras, reg_image_ids, fnames, pixels, pts3d, out_path);

  PLOGI << "================== Summary Begin ==========================";
  PLOGI << "Registered/Total: " << reg_image_ids.size() << '/' << fnames.size();
  PLOGI << "Error 2d-2d: " << error_2d2d;
  PLOGI << "Error 2d-3d: " << error_2d3d;
  PLOGI << "==================== Summary End ==========================";

  return 0;
}

cmdline::parser ParseArgs(int argc, char** argv)
{
  cmdline::parser parser;
  parser.add<string>("images", 'i', "Images directory", true, "");
  parser.add<string>("features", 'f', "Features and matches directory", true, "");
  parser.add<string>("annotation", 'a', "Annotation filepath", false, "");
  parser.add<string>("output", 'o', "Output directory", true, "");
  parser.add("dist", '\0', "Whether images have distortion");
  parser.parse_check(argc, argv);

  return parser;
}


bool RunPtzBA(const std::vector<std::string>& fnames, const std::vector<ptzcalib::ImageFeatures>& features,
              const std::vector<ptzcalib::MatchesInfo>& matches_info, int max_iter, std::vector<ptzcalib::Camera>& cameras,
              std::unordered_set<long>& reg_image_ids)
{
  size_t num_images = fnames.size();

  cameras.clear();
  cameras.resize(num_images);
  PtzIncrementalOptimizer ptz_iba(features, matches_info, cameras, fnames, max_iter);

  reg_image_ids.clear();
  bool ba_success = ptz_iba.Solve(cameras, reg_image_ids);

  return ba_success;
}

bool RunGeoreferencing(const std::vector<ptzcalib::ImageFeatures>& features, const std::vector<ptzcalib::MatchesInfo>& matches_info,
                       const std::vector<std::vector<cv::Point2f>>& pixels, const std::vector<std::vector<cv::Point3d>>& pts3d,
                       const std::unordered_set<long>& cam_ids, int max_iter, bool has_dist, std::vector<ptzcalib::Camera>& cameras,
                       double& error_2d2d, double& error_2d3d)
{
  FACTOR_TYPE factor_type;
  if (has_dist)
    factor_type = PTZRayDist;
  else
    factor_type = PTZRay;

  PTZRayOptimizer ptzray_optimizer(features, matches_info, cameras, pixels, pts3d, cam_ids, max_iter, factor_type);

  vector<vector<Ray>> rays;
  bool ba_success2 = ptzray_optimizer.Solve(cameras, rays);
  if (!ba_success2) {
    error_2d2d = error_2d3d = -1;
    return false;
  }

  error_2d2d = ptzray_optimizer.final_reproj_error_2d2d();
  error_2d3d = ptzray_optimizer.final_reproj_error_2d3d();

  return true;
}