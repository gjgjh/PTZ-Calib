/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 19:53:41
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-02-18 14:09:51
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#include "ptz_incremental_optimizer.h"

#include <iostream>
#include <limits>

#include "data_io.h"
#include "krt_optimizer.h"
#include "ptzray_optimizer.h"
#include "utils/logging.h"

using namespace std;

namespace ptzcalib {

long PtzIncrementalOptimizer::kMaxNumImages = 100000;
float PtzIncrementalOptimizer::kBaGlobalImagesRatio = 1.1;

PtzIncrementalOptimizer::PtzIncrementalOptimizer(const vector<ImageFeatures>& features, const vector<MatchesInfo>& matches_info,
                                                 const vector<Camera>& cameras, int max_iter)
    : cameras_(cameras), features_(features), matches_info_(matches_info), max_iter_(max_iter)
{
}

PtzIncrementalOptimizer::PtzIncrementalOptimizer(const vector<ImageFeatures>& features, const vector<MatchesInfo>& matches_info,
                                                 const vector<Camera>& cameras, const std::vector<std::string>& names, int max_iter)
    : cameras_(cameras), features_(features), matches_info_(matches_info), names_(names), max_iter_(max_iter)
{
}

bool PtzIncrementalOptimizer::Solve(vector<Camera>& cameras, std::unordered_set<long>& reg_image_ids)
{
  if (!CheckValid())
    return false;

  const int kInitNumTrials = 50;
  for (int num_trials = 0; num_trials < kInitNumTrials; ++num_trials) {
    // Register initial pair
    long image_id1, image_id2;
    bool find_init_success = FindInitialImagePair(image_id1, image_id2);
    if (!find_init_success) {
      PLOGI << "No good initial image pair found";
      return false;
    }

    PLOGI << "Initializing with image pair #" << image_id1 << " and #" << image_id2;

    bool reg_init_success = RegisterInitialImagePair(image_id1, image_id2);
    if (!reg_init_success) {
      PLOGI << "Initialization failed - possible solutions:" << endl
            << "     - try to relax the initialization constraints" << endl
            << "     - manually select an initial image pair";
      continue;
    }
    else {
      PLOGI << "Initialization success. Focal: " << cameras_[image_id1].K().at<double>(0, 0) << " , "
            << cameras_[image_id2].K().at<double>(0, 0);
    }

    AdjustGlobalBundle();

    size_t ba_prev_num_reg_images = NumRegImages();

    bool reg_next_success = true;
    while (reg_next_success) {
      reg_next_success = false;

      const auto next_image_ids = FindNextImages();

      if (next_image_ids.empty()) {
        break;
      }

      for (size_t reg_trial = 0; reg_trial < next_image_ids.size(); ++reg_trial) {
        const auto image_id = next_image_ids[reg_trial];
        reg_next_success = RegisterNextImage(image_id);

        PLOGI << "Register image #" << image_id << (reg_next_success ? " success" : " failed")
              << ", focal: " << cameras_[image_id].K().at<double>(0, 0) << ". Currently registered: " << NumRegImages()
              << ", total: " << features_.size();

        if (reg_next_success) {
          if (NumRegImages() >= kBaGlobalImagesRatio * ba_prev_num_reg_images) {
            bool gba_success = AdjustGlobalBundle();

            if (gba_success) {
              ba_prev_num_reg_images = NumRegImages();
              break;
            }
            else {
              reg_image_ids_.erase(image_id);  // TODO: erase all cameras which not pass global ba
              reg_next_success = false;
            }
          }
        }

        if (!reg_next_success) {
          PLOGI << "Could not register, trying another image";

          // If initial pair fails to continue for some time,
          // abort and try different initial pair.
          const long kMinNumInitialRegTrials = 30;
          const int kMinModelSize = 3;
          if (reg_trial >= kMinNumInitialRegTrials && NumRegImages() < static_cast<size_t>(kMinModelSize)) {
            break;
          }
        }
      }
    }

    AdjustGlobalBundle();

    reg_image_ids = reg_image_ids_;
    cameras = cameras_;

    return true;
  }
}

void PtzIncrementalOptimizer::SetSeedImageId(const std::vector<long>& image_ids)
{
  PLOGI << "Manually set seed image ids";
  seed_image_ids_ = image_ids;
}

bool PtzIncrementalOptimizer::CheckValid()
{
  if (features_.empty() || features_.size() != cameras_.size() || max_iter_ <= 0)
    return false;
  else
    return true;
}

bool PtzIncrementalOptimizer::FindInitialImagePair(long& image_id1, long& image_id2)
{
  vector<long> image_ids1;
  if (seed_image_ids_.empty()) {
    image_ids1 = FindFirstInitialImage();
  }
  else {
    image_ids1 = seed_image_ids_;
  }

  for (long i1 = 0; i1 < image_ids1.size(); ++i1) {
    image_id1 = image_ids1[i1];

    const vector<long> image_ids2 = FindSecondInitialImage(image_id1);

    for (long i2 = 0; i2 < image_ids2.size(); ++i2) {
      image_id2 = image_ids2[i2];

      const long pair_id = ImagePairToPairId(image_id1, image_id2);

      // Try every pair only once.
      if (init_image_pairs_.count(pair_id) > 0) {
        continue;
      }

      init_image_pairs_.insert(pair_id);

      return true;
    }
  }

  image_id1 = numeric_limits<long>::max();
  image_id2 = numeric_limits<long>::max();
  return false;
}

vector<long> PtzIncrementalOptimizer::FindFirstInitialImage() const
{
  size_t num_imgs = features_.size();
  vector<float> confidences_rank(num_imgs, 0.0f);

  for (const auto& match_info : matches_info_) {
    long src_idx = match_info.src_img_idx;
    long dst_idx = match_info.dst_img_idx;
    float confidence = match_info.confidence;

    confidences_rank[src_idx] += confidence;
    confidences_rank[dst_idx] += confidence;
  }

  vector<long> indices(num_imgs);
  iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(), [&](int A, int B) -> bool { return confidences_rank[A] > confidences_rank[B]; });

  vector<long> sorted_images_ids;
  for (auto& index : indices) {
    if (confidences_rank[index] <= 0.0f)
      break;
    sorted_images_ids.push_back(index);
  }

  return sorted_images_ids;
}

vector<long> PtzIncrementalOptimizer::FindSecondInitialImage(long image_id1) const
{
  size_t num_imgs = features_.size();
  vector<float> confidences_rank(num_imgs, 0.0f);

  for (const auto& match_info : matches_info_) {
    long src_idx = match_info.src_img_idx;
    long dst_idx = match_info.dst_img_idx;
    float confidence = match_info.confidence;
    const auto& matches = match_info.matches;

    const float kMinPixelDiff = 50;
    if (matches.empty())
      continue;
    if (image_id1 != src_idx && image_id1 != dst_idx)
      continue;
    if (image_id1 == src_idx && image_id1 == dst_idx)
      continue;
    if (CalPixelDiff(src_idx, dst_idx, matches) < kMinPixelDiff)
      continue;
    if (image_id1 == src_idx && image_id1 != dst_idx)
      confidences_rank[dst_idx] += confidence;
    if (image_id1 != src_idx && image_id1 == dst_idx)
      confidences_rank[src_idx] += confidence;
  }

  vector<long> indices(num_imgs);
  iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(), [&](int A, int B) -> bool { return confidences_rank[A] > confidences_rank[B]; });

  vector<long> sorted_images_ids;
  for (auto& index : indices) {
    if (confidences_rank[index] <= 0.0f)
      break;
    sorted_images_ids.push_back(index);
  }

  return sorted_images_ids;
}

std::vector<long> PtzIncrementalOptimizer::FindNextImages() const
{
  size_t num_imgs = features_.size();
  vector<float> confidences_rank(num_imgs, 0.0f);

  for (const auto& match_info : matches_info_) {
    long src_idx = match_info.src_img_idx;
    long dst_idx = match_info.dst_img_idx;
    float confidence = match_info.confidence;

    if (src_idx == dst_idx)
      continue;
    if (match_info.H.empty())
      continue;

    // Only try registration for a certain maximum number of times.
    const size_t kMaxRegTrials = 4;
    if (num_reg_trials_.count(src_idx) != 0 && num_reg_trials_.at(src_idx) > kMaxRegTrials)
      continue;
    if (num_reg_trials_.count(dst_idx) != 0 && num_reg_trials_.at(dst_idx) > kMaxRegTrials)
      continue;

    if (reg_image_ids_.find(src_idx) != reg_image_ids_.end() && reg_image_ids_.find(dst_idx) != reg_image_ids_.end()) {
      // Skip images that are already registered.
      continue;
    }
    else if (reg_image_ids_.find(src_idx) == reg_image_ids_.end() && reg_image_ids_.find(dst_idx) == reg_image_ids_.end()) {
      // Skip images that are not neighbor.
      continue;
    }
    else if (reg_image_ids_.find(src_idx) != reg_image_ids_.end() && reg_image_ids_.find(dst_idx) == reg_image_ids_.end()) {
      confidences_rank[dst_idx] += confidence;
    }
    else if (reg_image_ids_.find(src_idx) == reg_image_ids_.end() && reg_image_ids_.find(dst_idx) != reg_image_ids_.end()) {
      confidences_rank[src_idx] += confidence;
    }
  }

  vector<long> indices(num_imgs);
  iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(), [&](int A, int B) -> bool { return confidences_rank[A] > confidences_rank[B]; });

  vector<long> sorted_images_ids;
  for (auto& index : indices) {
    if (confidences_rank[index] <= 0.0f)
      break;
    sorted_images_ids.push_back(index);
  }

  return sorted_images_ids;
}

float PtzIncrementalOptimizer::CalPixelDiff(long image_id1, long image_id2, const vector<cv::DMatch>& matches) const
{
  size_t num_matches = matches.size();
  float total_dists = 0.0f;

  for (size_t i = 0; i < num_matches; ++i) {
    const cv::DMatch& m = matches[i];
    auto pt1 = features_[image_id1].keypoints[m.queryIdx].pt;
    auto pt2 = features_[image_id2].keypoints[m.trainIdx].pt;

    total_dists += cv::norm(pt1 - pt2);
  }

  return total_dists * 1.0f / num_matches;
}

long PtzIncrementalOptimizer::ImagePairToPairId(long image_id1, long image_id2) const
{
  if (image_id1 < image_id2)
    return image_id1 * kMaxNumImages + image_id2;
  else
    return image_id2 * kMaxNumImages + image_id1;
}

void PtzIncrementalOptimizer::SetInitialImagePairParameters(long image_id1, long image_id2)
{
  constexpr double ratio = 1.2;  // this affects fov
  double focal = ratio * max(features_[image_id1].img_size.width, features_[image_id1].img_size.height);
  double cx = 0.5 * features_[image_id1].img_size.width;
  double cy = 0.5 * features_[image_id1].img_size.height;
  cameras_[image_id1].K().at<double>(0, 0) = cameras_[image_id1].K().at<double>(1, 1) = focal;
  cameras_[image_id1].K().at<double>(0, 2) = cx;
  cameras_[image_id1].K().at<double>(1, 2) = cy;
  cameras_[image_id1].R() = cv::Mat_<double>::eye(3, 3);

  focal = ratio * max(features_[image_id2].img_size.width, features_[image_id2].img_size.height);
  cx = 0.5 * features_[image_id2].img_size.width;
  cy = 0.5 * features_[image_id2].img_size.height;
  cameras_[image_id2].K().at<double>(0, 0) = cameras_[image_id2].K().at<double>(1, 1) = focal;
  cameras_[image_id2].K().at<double>(0, 2) = cx;
  cameras_[image_id2].K().at<double>(1, 2) = cy;

  for (const auto& match_info : matches_info_) {
    long i = match_info.src_img_idx;
    long j = match_info.dst_img_idx;

    if (i == image_id1 && j == image_id2) {
      cv::Mat H_j_i = match_info.H;
      cv::Mat R_j_i = cameras_[j].K().inv() * H_j_i * cameras_[i].K();

      cameras_[j].R() = R_j_i * cameras_[i].R();
      break;
    }
  }
}

bool PtzIncrementalOptimizer::RegisterInitialImagePair(long image_id1, long image_id2)
{
  assert(NumRegImages() == 0);

  num_reg_trials_[image_id1] += 1;
  num_reg_trials_[image_id2] += 1;

  long pair_id = ImagePairToPairId(image_id1, image_id2);
  init_image_pairs_.insert(pair_id);

  SetInitialImagePairParameters(image_id1, image_id2);

  PTZRayOptimizer optimizer(features_, matches_info_, cameras_, {image_id1, image_id2}, max_iter_, PTZRay);
  bool ba_success = optimizer.Solve(cameras_);

  if (ba_success) {
    reg_image_ids_.insert(image_id1);
    reg_image_ids_.insert(image_id2);
  }

  return ba_success;
}

bool PtzIncrementalOptimizer::RegisterNextImage(long image_id)
{
  assert(NumRegImages() >= 2);
  assert(reg_image_ids_.find(image_id) == reg_image_ids_.end() && "Image cannot be registered multiple times");

  num_reg_trials_[image_id] += 1;

  for (const auto& match_info : matches_info_) {
    long i = match_info.src_img_idx;
    long j = match_info.dst_img_idx;
    cv::Mat H_j_i = match_info.H;
    if (H_j_i.empty())
      continue;

    if (reg_image_ids_.count(i) == 1 && j == image_id) {
      cameras_[j].K() = cameras_[i].K();
      cv::Mat R_j_i = cameras_[j].K().inv() * H_j_i * cameras_[i].K();
      cameras_[j].R() = R_j_i * cameras_[i].R();

      constexpr int max_iter = 100;
      constexpr double max_reproj_error = 100;
      KRTOptimizer optimizer(max_iter, max_reproj_error, KRTOptimizer::F);
      optimizer.SetInitParams(cameras_[j].K(), cameras_[j].R(), cameras_[j].t(), cameras_[j].dist());
      optimizer.Add2d2dConstraints(cameras_[i], features_[i].keypoints, features_[j].keypoints, match_info.matches);

      cv::Mat K, R, t, dist;
      bool ba_success = optimizer.Solve(K, R, t, dist);
      if (ba_success) {
        cameras_[j].K() = K;
        cameras_[j].R() = R;
        reg_image_ids_.insert(j);

        return true;
      }
      else {
        continue;
      }
    }
  }

  return false;
}

bool PtzIncrementalOptimizer::AdjustGlobalBundle()
{
  PLOGI << "Global bundle adjustment start";

  PTZRayOptimizer optimizer(features_, matches_info_, cameras_, reg_image_ids_, max_iter_, PTZRay);

  bool ba_success = optimizer.Solve(cameras_);
  double reprojection_error = optimizer.final_reproj_error_all();

  if (ba_success) {
    vector<long> reg_image_ids_sorted(reg_image_ids_.begin(), reg_image_ids_.end());
    sort(reg_image_ids_sorted.begin(), reg_image_ids_sorted.end());

    PLOGI << "Global bundle adjustment success. Rreprojecion error: " << reprojection_error;
  }
  else {
    PLOGI << "Global bundle adjustment failed. Rreprojecion error: " << reprojection_error;
  }

  return ba_success;
}

}  // namespace ptzcalib