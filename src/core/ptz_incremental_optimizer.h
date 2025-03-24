/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 19:53:37
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-02-18 14:10:13
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#ifndef PTZ_CALIB_SRC_CORE_PTZ_INCREMENTAL_OPTIMIZER_H
#define PTZ_CALIB_SRC_CORE_PTZ_INCREMENTAL_OPTIMIZER_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "types.h"

namespace ptzcalib {

class PtzIncrementalOptimizer {
 public:
  PtzIncrementalOptimizer(const std::vector<ImageFeatures>& features, const std::vector<MatchesInfo>& matches_info,
                          const std::vector<Camera>& cameras, int max_iter);
  PtzIncrementalOptimizer(const std::vector<ImageFeatures>& features, const std::vector<MatchesInfo>& matches_info,
                          const std::vector<Camera>& cameras, const std::vector<std::string>& names, int max_iter);
  ~PtzIncrementalOptimizer() = default;

  /** Run this optimizer
   * @param cameras output cameras
   * @param reg_image_ids successful registered image ids
   * @return success or not
   */
  bool Solve(std::vector<Camera>& cameras, std::unordered_set<long>& reg_image_ids);

  /** Manually set seed images ids
   * @param image_ids seed images ids
   */
  void SetSeedImageId(const std::vector<long>& image_ids);

 public:
  static long kMaxNumImages;
  static float kBaGlobalImagesRatio;

 private:
  bool CheckValid();

  /** Find initial image pair to seed the incremental reconstruction.
   * The image pairs should be passed to `RegisterInitialImagePair`.This
   * function automatically ignores image pairs that failed to register previously.
   * @param image_id1 image pair id1
   * @param image_id2 image pair id2
   * @return success or not
   */
  bool FindInitialImagePair(long& image_id1, long& image_id2);

  /** Find seed images for incremental reconstruction. Suitable seed images have a large number of correspondences and have larger matching
   * confidence. The returned list is ordered such that most suitable images are in the front.
   * @return first seed images id
   */
  std::vector<long> FindFirstInitialImage() const;

  /** For a given first seed image, find other images that are connected to the first image. Suitable second images have a large number of
   * correspondences to the first image and have large disparity. The returned list is ordered such that most suitable images are
   * in the front.
   * @param image_id1 input first seed image id
   * @return second seed images id
   */
  std::vector<long> FindSecondInitialImage(long image_id1) const;

  /** Find best next image to register in the incremental reconstruction. The images should be passed to `RegisterNextImage`.
   * @return next candidate images id
   */
  std::vector<long> FindNextImages() const;

  /** Calculate pixel disparity between two images
   * @param image_id1 image1 id
   * @param image_id2 image2 id
   * @param matches matches between two images
   * @return mean pixel disparity
   */
  float CalPixelDiff(long image_id1, long image_id2, const std::vector<cv::DMatch>& matches) const;

  /** Attempt to seed the reconstruction from an image pair.
   * @param image_id1 image1 id
   * @param image_id2 image2 id
   * @return register success or not
   */
  bool RegisterInitialImagePair(long image_id1, long image_id2);

  /** Attempt to register image to the existing model. This requires that a previous call to `RegisterInitialImagePair` was successful.
   * @param image_id register image id
   * @return register success or not
   */
  bool RegisterNextImage(long image_id);

  void SetInitialImagePairParameters(long image_id1, long image_id2);

  bool AdjustGlobalBundle();

  long ImagePairToPairId(long image_id1, long image_id2) const;

  size_t NumRegImages() const { return reg_image_ids_.size(); }

 private:
  std::vector<Camera> cameras_;
  std::vector<ImageFeatures> features_;
  std::vector<MatchesInfo> matches_info_;
  std::vector<std::string> names_;
  int max_iter_;

  // Images and image pairs that have been used for initialization. Each image
  // and image pair is only tried once for initialization.
  std::unordered_set<long> init_image_pairs_;

  // Number of trials to register image in current reconstruction. Used to set
  // an upper bound to the number of trials to register an image.
  std::unordered_map<long, size_t> num_reg_trials_;

  std::unordered_set<long> reg_image_ids_;

  std::vector<long> seed_image_ids_;
};

}  // namespace ptzcalib

#endif  // PTZ_CALIB_SRC_CORE_PTZ_INCREMENTAL_OPTIMIZER_H