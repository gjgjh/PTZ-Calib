/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 20:06:16
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-02-18 10:26:26
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#ifndef PTZ_CALIB_SRC_CORE_KRT_OPTIMIZER_H
#define PTZ_CALIB_SRC_CORE_KRT_OPTIMIZER_H

#include <ceres/ceres.h>

#include <opencv2/core.hpp>
#include <vector>

#include "types.h"

namespace ptzcalib {

/////////////////////////////////////// Factor2d2d ////////////////////////////////////////////////
// f, R
class Factor2d2d {
 public:
  Factor2d2d(const Camera& cam1, const cv::Point2f& uv1, const cv::Point2f& uv2) : cam1_(cam1), uv1_(uv1), uv2_(uv2) {}
  bool operator()(const double* const camera, double* residual) const;

  static ceres::CostFunction* Create(const Camera& cam1, const cv::Point2f& uv1, const cv::Point2f& uv2);

 private:
  cv::Point2f uv1_, uv2_;
  Camera cam1_;
};

/////////////////////////////////////// Factor2d2dFxfy ////////////////////////////////////////////////
// fx, fy, R
class Factor2d2dFxfy {
 public:
  Factor2d2dFxfy(const Camera& cam1, const cv::Point2f& uv1, const cv::Point2f& uv2) : cam1_(cam1), uv1_(uv1), uv2_(uv2) {}
  bool operator()(const double* const camera, double* residual) const;

  static ceres::CostFunction* Create(const Camera& cam1, const cv::Point2f& uv1, const cv::Point2f& uv2);

 private:
  cv::Point2f uv1_, uv2_;
  Camera cam1_;
};

/////////////////////////////////////// Factor2d2dDist ////////////////////////////////////////////////
// f, R, dist
class Factor2d2dDist {
 public:
  Factor2d2dDist(const Camera& cam1, const cv::Point2f& uv1, const cv::Point2f& uv2) : cam1_(cam1), uv1_(uv1), uv2_(uv2) {}
  bool operator()(const double* const camera, double* residual) const;

  static ceres::CostFunction* Create(const Camera& cam1, const cv::Point2f& uv1, const cv::Point2f& uv2);

 private:
  cv::Point2f uv1_, uv2_;
  Camera cam1_;
};

/////////////////////////////////////// Factor2d2dFxfyDist ////////////////////////////////////////////////
// fx, fy, R, dist
class Factor2d2dFxfyDist {
 public:
  Factor2d2dFxfyDist(const Camera& cam1, const cv::Point2f& uv1, const cv::Point2f& uv2) : cam1_(cam1), uv1_(uv1), uv2_(uv2) {}
  bool operator()(const double* const camera, double* residual) const;

  static ceres::CostFunction* Create(const Camera& cam1, const cv::Point2f& uv1, const cv::Point2f& uv2);

 private:
  cv::Point2f uv1_, uv2_;
  Camera cam1_;
};

/////////////////////////////////////// Factor2d3dDist ////////////////////////////////////////////////
// f, R, dist
class Factor2d3dDist {
 public:
  Factor2d3dDist(const cv::Point2f& pt2d, const cv::Point3d& pt3d) : pt2d_(pt2d), pt3d_(pt3d) {}
  bool operator()(const double* const camera, double* residual) const;

  static ceres::CostFunction* Create(const cv::Point2f& pt2d, const cv::Point3d& pt3d);

 private:
  cv::Point2f pt2d_;
  cv::Point3d pt3d_;
};

/////////////////////////////////////// Factor2d3dFxfyDist ////////////////////////////////////////////////
// fx, fy, R, dist
class Factor2d3dFxfyDist {
 public:
  Factor2d3dFxfyDist(const cv::Point2f& pt2d, const cv::Point3d& pt3d) : pt2d_(pt2d), pt3d_(pt3d) {}
  bool operator()(const double* const camera, double* residual) const;

  static ceres::CostFunction* Create(const cv::Point2f& pt2d, const cv::Point3d& pt3d);

 private:
  cv::Point2f pt2d_;
  cv::Point3d pt3d_;
};

/////////////////////////////////////// KRTOptimizer ////////////////////////////////////////////////

class KRTOptimizer {
 public:
  enum FACTOR_TYPE { F, FDist, Fxfy, FxfyDist };

  KRTOptimizer(int max_iter, double max_reproj_error, FACTOR_TYPE factor_type);
  ~KRTOptimizer() = default;

  void SetInitParams(const cv::Mat& K, const cv::Mat& R, const cv::Mat& t, const cv::Mat& dist);
  void Add2d2dConstraints(const Camera& cam_ref, const std::vector<cv::KeyPoint>& kpts_ref, const std::vector<cv::KeyPoint>& kpts_curr,
                          const std::vector<cv::DMatch>& matches);
  void Add2d3dConstraints(const std::vector<cv::Point2f>& pts2d, const std::vector<cv::Point3d>& pts3d);
  bool Solve(cv::Mat& K, cv::Mat& R, cv::Mat& t, cv::Mat& dist);
  double Cal2d2dReprojError(const Camera& cam_ref, const std::vector<cv::KeyPoint>& kpts_ref, const std::vector<cv::KeyPoint>& kpts_curr,
                            const std::vector<cv::DMatch>& matches);
  double Cal2d3dReprojError(const std::vector<cv::Point2f>& pts2d, const std::vector<cv::Point3d>& pts3d);
  void SetFixedFocal();

  int num_iter_ = 0;

 private:
  bool CheckResults(const ceres::Solver::Summary& summary);
  void ObtainRefinedCameraParams(cv::Mat& K, cv::Mat& R, cv::Mat& t, cv::Mat& dist);

  Camera cam_curr_world_;
  Camera cam_curr_local_;
  std::vector<double> cam_curr_local_param_;

  cv::Mat R_local_world_;
  cv::Mat t_local_world_;

  bool set_fixed_focal_ = false;

  FACTOR_TYPE factor_type_ = F;

  ceres::Problem problem_;
  int max_iter_ = 100;
  double max_reproj_error_ = 50;
};

}  // namespace ptzcalib

#endif  // PTZ_CALIB_SRC_CORE_KRT_OPTIMIZER_H