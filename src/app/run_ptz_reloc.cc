/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-03-11 14:50:09
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-03-24 15:44:40
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#include "run_ptz_reloc.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "core/data_io.h"
#include "core/krt_optimizer.h"
#include "run_ptz_ba.h"
#include "utils/logging.h"
#include "utils/os_path.h"

using namespace std;
using namespace ptzcalib;

int main(int argc, char** argv)
{
  InitLogging(false);

  cmdline::parser parser = ParseArgs(argc, argv);

  // Load images, features and matches
  vector<string> ref_fnames;
  vector<ImageFeatures> ref_features;
  vector<cv::Size> ref_sizes;
  bool load_status =
      LoadImgsAndFeatures(parser.get<string>("ref_images"), parser.get<string>("ref_features"), ref_fnames, ref_features, ref_sizes);
  if (!load_status) {
    PLOGE << "Error loading reference images and features. Exiting ...";
    return -1;
  }

  vector<string> test_fnames;
  vector<ImageFeatures> test_features;
  vector<cv::Size> test_sizes;
  load_status =
      LoadImgsAndFeatures(parser.get<string>("test_images"), parser.get<string>("test_features"), test_fnames, test_features, test_sizes);
  if (!load_status) {
    PLOGE << "Error loading test images and features. Exiting ...";
    return -1;
  }

  vector<vector<cv::DMatch>> pairs_matches;
  vector<pair<string, string>> img_pairs_name;
  string matches_path = parser.get<string>("test_features") + "/pairs_matches.txt";
  ReadColmapMatches(matches_path, pairs_matches, img_pairs_name);

  // Load reference camera parameters
  vector<Camera> ref_cameras;
  load_status = ReadCamFromJson(parser.get<string>("ref_params"), ref_fnames, ref_cameras);
  if (!load_status) {
    PLOGE << "Error loading reference camera parameters. Exiting ...";
    return -1;
  }

  // Run ptz-reloc for each test image
  vector<Camera> test_cameras(test_fnames.size());
  unordered_set<long> success_ids;

  for (size_t test_idx = 0; test_idx < test_fnames.size(); ++test_idx) {
    BestMatchT best_match = FindBestMatch(test_fnames[test_idx], img_pairs_name, pairs_matches);
    long ref_idx = FindImgIndex(ref_fnames, best_match.first);
    if (ref_idx == -1) {
      PLOGI << "Running ptz-reloc failed: " << test_fnames[test_idx];
      continue;
    }

    // {
    //   cv::Mat ref_img = cv::imread(parser.get<string>("ref_images") + "/" + ref_fnames[ref_idx]);
    //   cv::Mat test_img = cv::imread(parser.get<string>("test_images") + "/" + test_fnames[test_idx]);
    //   cv::Mat vis_matching =
    //       VisMatching(ref_img, ref_features[ref_idx].keypoints, test_img, test_features[test_idx].keypoints, best_match.second);

    //   if (!vis_matching.empty()) {
    //     string vis_name = basename(ref_fnames[ref_idx]) + "-" + basename(test_fnames[test_idx]) + ".jpg";
    //     cv::imwrite(vis_name, vis_matching);
    //   }
    // }

    Camera ref_cam = ref_cameras[ref_idx];

    static const int MAX_ITER = 200;
    static const double MAX_REPROJ_ERROR = 100.0;
    KRTOptimizer::FACTOR_TYPE factor_type = parser.exist("dist") ? KRTOptimizer::FDist : KRTOptimizer::F;

    KRTOptimizer optimizer(MAX_ITER, MAX_REPROJ_ERROR, factor_type);

    // Set initial parameter
    cv::Mat K, R, t, dist;
    double f = ref_cam.K().at<double>(0, 0);
    double cx = 0.5 * test_sizes[test_idx].width, cy = 0.5 * test_sizes[test_idx].height;
    K = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
    R = ref_cam.R();
    t = ref_cam.t();
    dist = ref_cam.dist();
    optimizer.SetInitParams(K, R, t, dist);

    optimizer.Add2d2dConstraints(ref_cam, ref_features[ref_idx].keypoints, test_features[test_idx].keypoints, best_match.second);

    bool opti_success = optimizer.Solve(K, R, t, dist);

    if (opti_success) {
      test_cameras[test_idx] = Camera(K, R, t, dist);
      success_ids.insert(test_idx);
      PLOGI << "Running ptz-reloc success: " << test_fnames[test_idx];
    }
    else {
      PLOGI << "Running ptz-reloc failed: " << test_fnames[test_idx];
    }
  }

  string cam_id = basename(parser.get<string>("test_images"));
  string out_dir = parser.get<string>("output");
  mkdir_ifnot_exist(out_dir);
  string out_path = out_dir + "/" + cam_id + ".json";

  vector<vector<cv::Point2f>> pixels(test_fnames.size());
  vector<vector<cv::Point3d>> pts3d(test_fnames.size());
  SaveRegisteredCam(test_cameras, success_ids, test_fnames, pixels, pts3d, out_path);

  return 0;
}

cmdline::parser ParseArgs(int argc, char** argv)
{
  cmdline::parser parser;
  parser.add<string>("ref_images", '\0', "Reference images directory", true, "");
  parser.add<string>("ref_features", '\0', "Reference images features directory", true, "");
  parser.add<string>("ref_params", '\0', "Reference camera parameters filepath", true, "");
  parser.add<string>("test_images", '\0', "Test images directory", true, "");
  parser.add<string>("test_features", '\0', "Test images features and matches directory", true, "");
  parser.add<string>("output", '\0', "Output directory", true, "");
  parser.add("dist", '\0', "Whether images have distortion");
  parser.parse_check(argc, argv);

  return parser;
}

BestMatchT FindBestMatch(const std::string& fname, const std::vector<std::pair<std::string, std::string>>& img_pairs_name,
                         const std::vector<std::vector<cv::DMatch>>& pairs_matches)
{
  if (img_pairs_name.size() != img_pairs_name.size()) {
    PLOGE << "Invalid input, size not matched: " << img_pairs_name.size() << " and " << pairs_matches.size();
    return BestMatchT();
  }

  BestMatchT best_match;

  for (size_t i = 0; i < img_pairs_name.size(); ++i) {
    if (img_pairs_name[i].second != fname)
      continue;

    if (pairs_matches[i].size() > best_match.second.size())
      best_match = {img_pairs_name[i].first, pairs_matches[i]};
  }

  return best_match;
}

cv::Mat VisMatching(const cv::Mat& img1, const std::vector<cv::KeyPoint>& kpts1, const cv::Mat& img2,
                    const std::vector<cv::KeyPoint>& kpts2, const std::vector<cv::DMatch>& matches12)
{
  if (img1.empty() || img2.empty()) {
    PLOGW << "Empty input images for visualizing";
    return cv::Mat();
  }

  // save only matched keypoints
  std::vector<cv::KeyPoint> kpts1_matched, kpts2_matched;
  std::vector<cv::DMatch> matches12_new;
  kpts1_matched.reserve(matches12.size());
  kpts2_matched.reserve(matches12.size());
  matches12_new.reserve(matches12.size());

  for (size_t i = 0; i < matches12.size(); ++i) {
    cv::DMatch m = matches12[i];
    kpts1_matched.push_back(kpts1[m.queryIdx]);
    kpts2_matched.push_back(kpts2[m.trainIdx]);
    m.queryIdx = m.trainIdx = i;
    matches12_new.push_back(m);
  }

  cv::Mat vis;
  cv::drawMatches(img1, kpts1_matched, img2, kpts2_matched, matches12_new, vis, cv::Scalar::all(-1), cv::Scalar::all(-1));
  cv::putText(vis, "Match numbers: " + to_string(matches12_new.size()), cv::Point(40, 100), cv::QT_FONT_NORMAL, 1.0, cv::Scalar(0, 0, 255),
              1.5);
  return vis;
}