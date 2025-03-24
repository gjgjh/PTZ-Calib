/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 20:06:06
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-02-18 10:15:40
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#ifndef PTZ_CALIB_SRC_CORE_PTZRAY_OPTIMIZER_H
#define PTZ_CALIB_SRC_CORE_PTZRAY_OPTIMIZER_H

#include <ceres/ceres.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tracks.h"
#include "types.h"

namespace ptzcalib {

/** Simplest PTZ camera model: f, R, ray */
class PTZRayFactor {
 public:
  PTZRayFactor(const cv::Point2f& uv) : uv_(uv) {}
  bool operator()(const double* const intrinsics, const double* const extrinsics, const double* const ray, double* residual) const;

  static ceres::CostFunction* Create(const cv::Point2f& uv);

 private:
  cv::Point2f uv_;
};

/** PTZ camera model with distortion: f, R, k1, k2, k3, p1, p2, ray
 * x = KRX
 */
class PTZRayDistFactor {
 public:
  PTZRayDistFactor(const cv::Point2f& uv) : uv_(uv) {}
  bool operator()(const double* const intrinsics, const double* const extrinsics, const double* const ray, double* residual) const;

  static ceres::CostFunction* Create(const cv::Point2f& uv);

 private:
  cv::Point2f uv_;
};

/** PTZ camera model with distortion: fx, fy, R, k1, k2, k3, p1, p2, ray
 * x = KRX
 */
class PTZRayFxfyDistFactor {
 public:
  PTZRayFxfyDistFactor(const cv::Point2f& uv) : uv_(uv) {}
  bool operator()(const double* const intrinsics, const double* const extrinsics, const double* const ray, double* residual) const;

  static ceres::CostFunction* Create(const cv::Point2f& uv);

 private:
  cv::Point2f uv_;
};

/** PTZ camera model with distortion and displacement: f, R, d1, d2, d3, k1, k2, k3, p1, p2, ray
 * displacement: offset between the projetion center and the rotation center
 * x = K[R|t]X, t is displacement here
 */
class PTZRayDistDispFactor {
 public:
  PTZRayDistDispFactor(const cv::Point2f& uv) : uv_(uv) {}
  bool operator()(const double* const intrinsics, const double* const disp, const double* const extrinsics, const double* const ray,
                  double* residual) const;

  static ceres::CostFunction* Create(const cv::Point2f& uv);

 private:
  cv::Point2f uv_;
};

/** 2d-3d reprojection error term */
class Reproj2d3dFactor {
 public:
  Reproj2d3dFactor(const cv::Point2f& uv, const cv::Point3d& pt3d) : uv_(uv), pt3d_(pt3d) {}
  bool operator()(const double* const intrinsics, const double* const extrinsics, const double* const tlw, double* residual) const;

  static ceres::CostFunction* Create(const cv::Point2f& uv, const cv::Point3d& pt3d);

 private:
  cv::Point2f uv_;
  cv::Point3d pt3d_;
};

/** 2d-3d reprojection error term with displacement
 * displacement: offset between the projetion center and the rotation center
 * x = K[R|t]X, t is displacement here
 */
class Reproj2d3dDispFactor {
 public:
  Reproj2d3dDispFactor(const cv::Point2f& uv, const cv::Point3d& pt3d) : uv_(uv), pt3d_(pt3d) {}
  bool operator()(const double* const intrinsics, const double* const disp, const double* const extrinsics, const double* const tlw,
                  double* residual) const;

  static ceres::CostFunction* Create(const cv::Point2f& uv, const cv::Point3d& pt3d);

 private:
  cv::Point2f uv_;
  cv::Point3d pt3d_;
};

enum FACTOR_TYPE { PTZRay, PTZRayDist, PTZRayFxfyDist, PTZRayDistDisp };

class PTZRayOptimizer {
 public:
  PTZRayOptimizer(const std::vector<ImageFeatures>& features, const std::vector<MatchesInfo>& matches_info,
                  const std::vector<Camera>& cameras, const std::vector<std::vector<cv::Point2f>>& pixels,
                  const std::vector<std::vector<cv::Point3d>>& pts3d, const std::unordered_set<long>& cam_ids, int max_iter,
                  FACTOR_TYPE type);
  PTZRayOptimizer(const std::vector<ImageFeatures>& features, const std::vector<MatchesInfo>& matches_info,
                  const std::vector<Camera>& cameras, const std::unordered_set<long>& cam_ids, int max_iter, FACTOR_TYPE type);
  ~PTZRayOptimizer() = default;
  bool Solve(std::vector<Camera>& cameras);
  bool Solve(std::vector<Camera>& cameras, std::vector<std::vector<Ray>>& rays);
  double final_reproj_error_all() const;
  double final_reproj_error_2d2d() const;
  double final_reproj_error_2d3d() const;
  void SetSharedIntrinsics(const std::vector<long>& shared_ic_ids);

  static void T_l_w(const double* const tlw, cv::Mat& R_l_w, cv::Mat& t_l_w);

 private:
  bool CheckValid() const;
  void FindTracks();
  bool isCandidate(long image_id) const;
  bool SetInitTransLocalToWorld();
  void SetUpInitialCameraParams();
  void ObtainRefinedCameraParams(std::vector<Camera>& cameras, std::vector<std::vector<Ray>>& rays);
  cv::Mat Pix2Ray(const std::vector<Camera>& cameras, const std::vector<ImageFeatures>& features, const Track& track) const;
  void AddConstraints2d2d();
  void AddConstraints2d3d();
  void CalReprojError();
  void CalReprojError2d2d();
  void CalReprojError2d3d();

 private:
  std::vector<Camera> cameras_;
  std::vector<ImageFeatures> features_;
  std::vector<MatchesInfo> matches_info_;
  std::vector<std::vector<cv::Point2f>> pixels_;
  std::vector<std::vector<cv::Point3d>> pts3d_;
  size_t num_cams_ = 0;

  std::unordered_set<long> cam_ids_;  //!< camera ids that to be optimized
  std::vector<long> shared_ic_ids_;   //!< shared intrinsic ids
  FACTOR_TYPE type_;

  cv::Mat T_l_w_;  //!< Transform matrix from world to local

  std::unordered_map<long, std::vector<double>> intrinsics_param_;  //!< fx, fy, cx, cy, k1, k2, k3, p1, p2
  std::unordered_map<long, std::vector<double>> extrinsics_param_;  //!< r1, r2, r3, t1, t2, t3
  std::unordered_map<int, std::vector<double>> rays_param_;         //!< x, y, z
  std::vector<double> disp_param_;                                  //!< d1, d2, d3
  std::vector<double> tlw_param_;                                   //!< T_l_w_ parameters: r1, r2, r3, t1, t2, t3

  Tracks tracks_;
  int track_len_ = 0;
  int max_track_len_ = 0;
  int min_track_len_ = 0;

  ceres::Problem problem_;
  ceres::Solver::Summary summary_;
  int max_iter_ = 100;

  double init_reproj_error_all_ = 0.0f;
  double final_reproj_error_all_ = 0.0f;
  double final_reproj_error_2d2d_ = 0.0f;
  double final_reproj_error_2d3d_ = 0.0f;
};

}  // namespace ptzcalib

#endif  // PTZ_CALIB_SRC_CORE_PTZRAY_OPTIMIZER_H