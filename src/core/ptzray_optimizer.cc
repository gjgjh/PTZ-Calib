/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 20:06:06
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-02-17 20:07:55
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#include "ptzray_optimizer.h"

#include "utils/logging.h"

using namespace std;

namespace ptzcalib {

/////////////////////////////////// PTZRayFactor /////////////////////////////////////

bool PTZRayFactor::operator()(const double* const intrinsics, const double* const extrinsics, const double* const ray,
                              double* residual) const
{
  vector<double> param(15);
  param[0] = intrinsics[0];
  param[1] = intrinsics[0];
  param[2] = intrinsics[2];
  param[3] = intrinsics[3];

  param[4] = extrinsics[0];
  param[5] = extrinsics[1];
  param[6] = extrinsics[2];
  param[7] = extrinsics[3];
  param[8] = extrinsics[4];
  param[9] = extrinsics[5];

  param[10] = intrinsics[4];
  param[11] = intrinsics[5];
  param[12] = intrinsics[6];
  param[13] = intrinsics[7];
  param[14] = intrinsics[8];

  Camera cam;
  cam.FromVector(param);

  cv::Mat cv_ray = (cv::Mat_<double>(3, 1) << ray[0], ray[1], ray[2]);
  cv_ray /= cv::norm(cv_ray);

  // x = KRX
  cv::Mat uv_predict = cam.K() * cam.R() * cv_ray;
  uv_predict /= uv_predict.at<double>(2, 0);

  residual[0] = uv_.x - uv_predict.at<double>(0, 0);
  residual[1] = uv_.y - uv_predict.at<double>(1, 0);

  return true;
}

ceres::CostFunction* PTZRayFactor::Create(const cv::Point2f& uv)
{
  return (new ceres::NumericDiffCostFunction<PTZRayFactor, ceres::CENTRAL, 2, 9, 6, 3>(new PTZRayFactor(uv)));
}

/////////////////////////////////// PTZRayDistFactor /////////////////////////////////////

bool PTZRayDistFactor::operator()(const double* const intrinsics, const double* const extrinsics, const double* const ray,
                                  double* residual) const
{
  vector<double> param(15);
  param[0] = intrinsics[0];
  param[1] = intrinsics[0];
  param[2] = intrinsics[2];
  param[3] = intrinsics[3];

  param[4] = extrinsics[0];
  param[5] = extrinsics[1];
  param[6] = extrinsics[2];
  param[7] = extrinsics[3];
  param[8] = extrinsics[4];
  param[9] = extrinsics[5];

  param[10] = intrinsics[4];
  param[11] = intrinsics[5];
  param[12] = intrinsics[6];
  param[13] = intrinsics[7];
  param[14] = intrinsics[8];

  Camera cam;
  cam.FromVector(param);

  cv::Mat cv_ray = (cv::Mat_<double>(3, 1) << ray[0], ray[1], ray[2]);
  // cv_ray /= cv::norm(cv_ray);

  // x = KRX
  cv::Mat pt3d = cam.R() * cv_ray;

  // add penalty if behind the camera
  const double kPenalty = 1000000.0;
  if (pt3d.at<double>(2, 0) < 0) {
    residual[0] = kPenalty;
    residual[1] = kPenalty;
    return true;
  }

  pt3d /= pt3d.at<double>(2, 0);
  double x = pt3d.at<double>(0, 0), y = pt3d.at<double>(1, 0);

  double fx = param[0], fy = param[1], cx = param[2], cy = param[3];
  double k1 = param[10], k2 = param[11], k3 = param[12];
  double p1 = param[13], p2 = param[14];

  double r2 = x * x + y * y;
  double r4 = r2 * r2;
  double r6 = r2 * r2 * r2;
  double xy = x * y;
  double x2 = x * x;
  double y2 = y * y;
  double radial_dist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

  double x_distorted = x * radial_dist + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2);
  double y_distorted = y * radial_dist + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2);

  double x_proj = fx * x_distorted + cx;
  double y_proj = fy * y_distorted + cy;

  residual[0] = uv_.x - x_proj;
  residual[1] = uv_.y - y_proj;

  return true;
}

ceres::CostFunction* PTZRayDistFactor::Create(const cv::Point2f& uv)
{
  return (new ceres::NumericDiffCostFunction<PTZRayDistFactor, ceres::CENTRAL, 2, 9, 6, 3>(new PTZRayDistFactor(uv)));
}

/////////////////////////////////// PTZRayFxfyDistFactor /////////////////////////////////////

bool PTZRayFxfyDistFactor::operator()(const double* const intrinsics, const double* const extrinsics, const double* const ray,
                                      double* residual) const
{
  vector<double> param(15);
  param[0] = intrinsics[0];
  param[1] = intrinsics[1];
  param[2] = intrinsics[2];
  param[3] = intrinsics[3];

  param[4] = extrinsics[0];
  param[5] = extrinsics[1];
  param[6] = extrinsics[2];
  param[7] = extrinsics[3];
  param[8] = extrinsics[4];
  param[9] = extrinsics[5];

  param[10] = intrinsics[4];
  param[11] = intrinsics[5];
  param[12] = intrinsics[6];
  param[13] = intrinsics[7];
  param[14] = intrinsics[8];

  Camera cam;
  cam.FromVector(param);

  cv::Mat cv_ray = (cv::Mat_<double>(3, 1) << ray[0], ray[1], ray[2]);
  cv_ray /= cv::norm(cv_ray);

  // x = KRX
  cv::Mat pt3d = cam.R() * cv_ray;
  pt3d /= pt3d.at<double>(2, 0);
  double x = pt3d.at<double>(0, 0), y = pt3d.at<double>(1, 0);

  double fx = param[0], fy = param[1], cx = param[2], cy = param[3];
  double k1 = param[10], k2 = param[11], k3 = param[12];
  double p1 = param[13], p2 = param[14];

  double r2 = x * x + y * y;
  double r4 = r2 * r2;
  double r6 = r2 * r2 * r2;
  double xy = x * y;
  double x2 = x * x;
  double y2 = y * y;
  double radial_dist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

  double x_distorted = x * radial_dist + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2);
  double y_distorted = y * radial_dist + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2);

  double x_proj = fx * x_distorted + cx;
  double y_proj = fy * y_distorted + cy;

  residual[0] = uv_.x - x_proj;
  residual[1] = uv_.y - y_proj;

  return true;
}

ceres::CostFunction* PTZRayFxfyDistFactor::Create(const cv::Point2f& uv)
{
  return (new ceres::NumericDiffCostFunction<PTZRayFxfyDistFactor, ceres::CENTRAL, 2, 9, 6, 3>(new PTZRayFxfyDistFactor(uv)));
}

/////////////////////////////////// PTZRayDistDispFactor /////////////////////////////////////

bool PTZRayDistDispFactor::operator()(const double* const intrinsics, const double* const disp, const double* const extrinsics,
                                      const double* const ray, double* residual) const
{
  vector<double> param(15);
  param[0] = intrinsics[0];
  param[1] = intrinsics[0];
  param[2] = intrinsics[2];
  param[3] = intrinsics[3];

  param[4] = extrinsics[0];
  param[5] = extrinsics[1];
  param[6] = extrinsics[2];
  param[7] = extrinsics[3];
  param[8] = extrinsics[4];
  param[9] = extrinsics[5];

  param[10] = intrinsics[4];
  param[11] = intrinsics[5];
  param[12] = intrinsics[6];
  param[13] = intrinsics[7];
  param[14] = intrinsics[8];

  Camera cam;
  cam.FromVector(param);

  cv::Mat cv_ray = (cv::Mat_<double>(3, 1) << ray[0], ray[1], ray[2]);
  cv_ray /= cv::norm(cv_ray);

  cv::Mat pt3d = cam.R() * cv_ray;
  double displacement = disp[0] + disp[1] * param[0] + disp[2] * param[0] * param[0];
  pt3d.at<double>(2, 0) += displacement;

  pt3d /= pt3d.at<double>(2, 0);
  double x = pt3d.at<double>(0, 0), y = pt3d.at<double>(1, 0);

  double fx = param[0], fy = param[1], cx = param[2], cy = param[3];
  double k1 = param[10], k2 = param[11], k3 = param[12];
  double p1 = param[13], p2 = param[14];

  double r2 = x * x + y * y;
  double r4 = r2 * r2;
  double r6 = r2 * r2 * r2;
  double xy = x * y;
  double x2 = x * x;
  double y2 = y * y;
  double radial_dist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

  double x_distorted = x * radial_dist + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2);
  double y_distorted = y * radial_dist + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2);

  double x_proj = fx * x_distorted + cx;
  double y_proj = fy * y_distorted + cy;

  residual[0] = uv_.x - x_proj;
  residual[1] = uv_.y - y_proj;

  return true;
}

ceres::CostFunction* PTZRayDistDispFactor::Create(const cv::Point2f& uv)
{
  return (new ceres::NumericDiffCostFunction<PTZRayDistDispFactor, ceres::CENTRAL, 2, 9, 3, 6, 3>(new PTZRayDistDispFactor(uv)));
}

/////////////////////////////////// Reproj2d3dFactor /////////////////////////////////////

bool Reproj2d3dFactor::operator()(const double* const intrinsics, const double* const extrinsics, const double* const tlw,
                                  double* residual) const
{
  vector<double> param(15);
  param[0] = intrinsics[0];
  param[1] = intrinsics[1];
  param[2] = intrinsics[2];
  param[3] = intrinsics[3];

  param[4] = extrinsics[0];
  param[5] = extrinsics[1];
  param[6] = extrinsics[2];
  param[7] = extrinsics[3];
  param[8] = extrinsics[4];
  param[9] = extrinsics[5];

  param[10] = intrinsics[4];
  param[11] = intrinsics[5];
  param[12] = intrinsics[6];
  param[13] = intrinsics[7];
  param[14] = intrinsics[8];

  Camera cam;
  cam.FromVector(param);

  cv::Mat R_l_w, t_l_w;
  PTZRayOptimizer::T_l_w(tlw, R_l_w, t_l_w);

  cv::Mat pt3d_w = (cv::Mat_<double>(3, 1) << pt3d_.x, pt3d_.y, pt3d_.z);
  cv::Mat pt3d_l = R_l_w * pt3d_w + t_l_w;

  // x = KRX
  cv::Mat pt_cam = cam.R() * pt3d_l;
  pt_cam /= pt_cam.at<double>(2, 0);
  double x = pt_cam.at<double>(0, 0), y = pt_cam.at<double>(1, 0);

  double fx = param[0], fy = param[1], cx = param[2], cy = param[3];
  double k1 = param[10], k2 = param[11], k3 = param[12];
  double p1 = param[13], p2 = param[14];

  double r2 = x * x + y * y;
  double r4 = r2 * r2;
  double r6 = r2 * r2 * r2;
  double xy = x * y;
  double x2 = x * x;
  double y2 = y * y;
  double radial_dist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

  double x_distorted = x * radial_dist + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2);
  double y_distorted = y * radial_dist + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2);

  double x_proj = fx * x_distorted + cx;
  double y_proj = fy * y_distorted + cy;

  residual[0] = uv_.x - x_proj;
  residual[1] = uv_.y - y_proj;

  return true;
}

ceres::CostFunction* Reproj2d3dFactor::Create(const cv::Point2f& uv, const cv::Point3d& pt3d)
{
  return (new ceres::NumericDiffCostFunction<Reproj2d3dFactor, ceres::CENTRAL, 2, 9, 6, 6>(new Reproj2d3dFactor(uv, pt3d)));
}

/////////////////////////////////// Reproj2d3dDispFactor ////////////////////////////////

bool Reproj2d3dDispFactor::operator()(const double* const intrinsics, const double* const disp, const double* const extrinsics,
                                      const double* const tlw, double* residual) const
{
  vector<double> param(15);
  param[0] = intrinsics[0];
  param[1] = intrinsics[1];
  param[2] = intrinsics[2];
  param[3] = intrinsics[3];

  param[4] = extrinsics[0];
  param[5] = extrinsics[1];
  param[6] = extrinsics[2];
  param[7] = extrinsics[3];
  param[8] = extrinsics[4];
  param[9] = extrinsics[5];

  param[10] = intrinsics[4];
  param[11] = intrinsics[5];
  param[12] = intrinsics[6];
  param[13] = intrinsics[7];
  param[14] = intrinsics[8];

  Camera cam;
  cam.FromVector(param);

  cv::Mat R_l_w, t_l_w;
  PTZRayOptimizer::T_l_w(tlw, R_l_w, t_l_w);

  cv::Mat pt3d_w = (cv::Mat_<double>(3, 1) << pt3d_.x, pt3d_.y, pt3d_.z);
  cv::Mat pt3d_l = R_l_w * pt3d_w + t_l_w;

  cv::Mat pt_cam = cam.R() * pt3d_l;

  double displacement = disp[0] + disp[1] * param[0] + disp[2] * param[0] * param[0];
  pt_cam.at<double>(2, 0) += displacement;

  pt_cam /= pt_cam.at<double>(2, 0);
  double x = pt_cam.at<double>(0, 0), y = pt_cam.at<double>(1, 0);

  double fx = param[0], fy = param[1], cx = param[2], cy = param[3];
  double k1 = param[10], k2 = param[11], k3 = param[12];
  double p1 = param[13], p2 = param[14];

  double r2 = x * x + y * y;
  double r4 = r2 * r2;
  double r6 = r2 * r2 * r2;
  double xy = x * y;
  double x2 = x * x;
  double y2 = y * y;
  double radial_dist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

  double x_distorted = x * radial_dist + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2);
  double y_distorted = y * radial_dist + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2);

  double x_proj = fx * x_distorted + cx;
  double y_proj = fy * y_distorted + cy;

  residual[0] = uv_.x - x_proj;
  residual[1] = uv_.y - y_proj;

  return true;
}

ceres::CostFunction* Reproj2d3dDispFactor::Create(const cv::Point2f& uv, const cv::Point3d& pt3d)
{
  return (new ceres::NumericDiffCostFunction<Reproj2d3dDispFactor, ceres::CENTRAL, 2, 9, 3, 6, 6>(new Reproj2d3dDispFactor(uv, pt3d)));
}

/////////////////////////////////// PTZRayOptimizer /////////////////////////////////////

PTZRayOptimizer::PTZRayOptimizer(const vector<ImageFeatures>& features, const vector<MatchesInfo>& matches_info,
                                 const vector<Camera>& cameras, const std::vector<std::vector<cv::Point2f>>& pixels,
                                 const std::vector<std::vector<cv::Point3d>>& pts3d, const unordered_set<long>& cam_ids, int max_iter,
                                 FACTOR_TYPE type)
    : features_(features),
      matches_info_(matches_info),
      cameras_(cameras),
      pixels_(pixels),
      pts3d_(pts3d),
      max_iter_(max_iter),
      type_(type),
      num_cams_(cameras.size())
{
  if (cam_ids.empty()) {
    vector<long> cam_ids_tmp(cameras_.size());
    iota(cam_ids_tmp.begin(), cam_ids_tmp.end(), 0);
    cam_ids_ = unordered_set<long>(cam_ids_tmp.begin(), cam_ids_tmp.end());
  }
  else {
    cam_ids_ = cam_ids;
  }

  shared_ic_ids_.resize(cameras_.size());
  iota(shared_ic_ids_.begin(), shared_ic_ids_.end(), 0);
}

PTZRayOptimizer::PTZRayOptimizer(const vector<ImageFeatures>& features, const vector<MatchesInfo>& matches_info,
                                 const vector<Camera>& cameras, const unordered_set<long>& cam_ids, int max_iter, FACTOR_TYPE type)
    : features_(features), matches_info_(matches_info), cameras_(cameras), max_iter_(max_iter), type_(type), num_cams_(cameras.size())
{
  if (cam_ids.empty()) {
    vector<long> cam_ids_tmp(cameras_.size());
    iota(cam_ids_tmp.begin(), cam_ids_tmp.end(), 0);
    cam_ids_ = unordered_set<long>(cam_ids_tmp.begin(), cam_ids_tmp.end());
  }
  else {
    cam_ids_ = cam_ids;
  }

  shared_ic_ids_.resize(cameras_.size());
  iota(shared_ic_ids_.begin(), shared_ic_ids_.end(), 0);
}

bool PTZRayOptimizer::Solve(std::vector<Camera>& cameras)
{
  std::vector<std::vector<Ray>> rays;
  return Solve(cameras, rays);
}

bool PTZRayOptimizer::Solve(vector<Camera>& cameras, std::vector<std::vector<Ray>>& rays)
{
  if (!CheckValid())
    return false;

  FindTracks();

  SetInitTransLocalToWorld();

  SetUpInitialCameraParams();

  AddConstraints2d2d();

  AddConstraints2d3d();

  ceres::Solver::Options options;
  options.max_num_iterations = max_iter_;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = 32;

  ceres::Solve(options, &problem_, &summary_);

  CalReprojError();

  PLOGI << summary_.BriefReport();
  PLOGI << "Init reprojection error all: " << init_reproj_error_all_ << ", final reprojection error all: " << final_reproj_error_all_;

  if (summary_.termination_type == ceres::TerminationType::CONVERGENCE) {
    ObtainRefinedCameraParams(cameras, rays);
    return true;
  }
  else {
    return false;
  }
}

double PTZRayOptimizer::final_reproj_error_all() const { return final_reproj_error_all_; }

double PTZRayOptimizer::final_reproj_error_2d2d() const { return final_reproj_error_2d2d_; }

double PTZRayOptimizer::final_reproj_error_2d3d() const { return final_reproj_error_2d3d_; }

void PTZRayOptimizer::SetSharedIntrinsics(const std::vector<long>& shared_ic_ids)
{
  if (shared_ic_ids.size() != cameras_.size()) {
    PLOGW << "Set shared intrinsics failed, length not matched: " << cameras_.size() << " - " << shared_ic_ids.size();
    return;
  }

  shared_ic_ids_ = shared_ic_ids;
}

void PTZRayOptimizer::T_l_w(const double* const tlw, cv::Mat& R_l_w, cv::Mat& t_l_w)
{
  cv::Mat rvec = (cv::Mat_<double>(3, 1) << tlw[0], tlw[1], tlw[2]);
  cv::Rodrigues(rvec, R_l_w);

  t_l_w = (cv::Mat_<double>(3, 1) << tlw[3], tlw[4], tlw[5]);
}

bool PTZRayOptimizer::CheckValid() const
{
  if (num_cams_ == 0)
    return false;
  if (features_.size() != num_cams_)
    return false;
  if (max_iter_ <= 0)
    return false;

  if (!pixels_.empty()) {
    if (pixels_.size() != num_cams_ || pts3d_.size() != num_cams_)
      return false;

    for (size_t i = 0; i < num_cams_; ++i) {
      if (pixels_[i].size() != pts3d_[i].size())
        return false;
    }
  }

  return true;
}

void PTZRayOptimizer::FindTracks()
{
  TracksBuilder builder;
  builder.Build(matches_info_);
  builder.Filter(4);
  builder.ExportToSTL(tracks_);

  Length(tracks_, track_len_, max_track_len_, min_track_len_);
  double mean_track_length = track_len_ / static_cast<double>(tracks_.size());
  PLOGI << "Tracks number: " << tracks_.size() << ", total track length: " << track_len_ << ", mean track length: " << mean_track_length
        << ", min track length: " << min_track_len_ << ", max track length: " << max_track_len_;

  set<int> max_connect_imgs;
  FindMaxCoVisible(tracks_, num_cams_, max_connect_imgs);
  PLOGI << "Max co-visible number: " << max_connect_imgs.size() << ", total number: " << num_cams_;
}

bool PTZRayOptimizer::isCandidate(long image_id) const
{
  if (cam_ids_.find(image_id) == cam_ids_.end())
    return false;
  else
    return true;
}

bool PTZRayOptimizer::SetInitTransLocalToWorld()
{
  for (size_t i = 0; i < num_cams_; ++i) {
    if (!isCandidate(i))
      continue;
    if (pixels_.empty() || pixels_[i].empty())
      continue;

    // Calculate T_i_w by PnP
    cv::Mat rvec, tvec;
    bool pnp_success = cv::solvePnP(pts3d_[i], pixels_[i], cameras_[i].K(), cameras_[i].dist(), rvec, tvec, false, cv::SOLVEPNP_EPNP);
    if (!pnp_success) {
      PLOGW << "SolvePnP failure";
      continue;
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    cv::Mat p3d = (cv::Mat_<double>(3, 1) << pts3d_[i][0].x, pts3d_[i][0].y, pts3d_[i][0].z);
    p3d = R * p3d + tvec;
    if (p3d.at<double>(2, 0) < 0 || cv::determinant(R) < 0.0f) {
      PLOGW << "SolvePnP failure";
      continue;
    }

    vector<cv::Point3f> pts3d_float;
    pts3d_float.reserve(pts3d_[i].size());
    for (size_t j = 0; j < pts3d_[i].size(); ++j) {
      pts3d_float.emplace_back(float(pts3d_[i][j].x), float(pts3d_[i][j].y), float(pts3d_[i][j].z));
    }

    std::vector<cv::Point2f> predict_pixels;
    cv::projectPoints(pts3d_float, rvec, tvec, cameras_[i].K(), cv::Mat(), predict_pixels);
    double reproj_error = 0;
    for (size_t j = 0; j < predict_pixels.size(); ++j) {
      reproj_error += (predict_pixels[j].x - pixels_[i][j].x) * (predict_pixels[j].x - pixels_[i][j].x) +
                      (predict_pixels[j].y - pixels_[i][j].y) * (predict_pixels[j].y - pixels_[i][j].y);
    }
    reproj_error = sqrt(reproj_error / predict_pixels.size());
    if (reproj_error > 300) {
      PLOGW << "Init reprojection error too large: " << reproj_error;
      continue;
    }
    else {
      PLOGI << "Init reprojection error: " << reproj_error;
    }

    cv::Mat T_i_w = cv::Mat_<double>::eye(4, 4);
    R.copyTo(T_i_w(cv::Range(0, 3), cv::Range(0, 3)));
    tvec.copyTo(T_i_w(cv::Range(0, 3), cv::Range(3, 4)));

    // T_l_w = T_i_l^{-1} * T_i_w
    cv::Mat T_i_l = cv::Mat_<double>::eye(4, 4);
    cameras_[i].R().copyTo(T_i_l(cv::Range(0, 3), cv::Range(0, 3)));
    cameras_[i].t().copyTo(T_i_l(cv::Range(0, 3), cv::Range(3, 4)));

    T_l_w_ = T_i_l.inv() * T_i_w;
    cv::Mat R_l_w, rvec_l_w, t_l_w;
    T_l_w_(cv::Range(0, 3), cv::Range(0, 3)).copyTo(R_l_w);
    T_l_w_(cv::Range(0, 3), cv::Range(3, 4)).copyTo(t_l_w);
    cv::Rodrigues(R_l_w, rvec_l_w);

    tlw_param_ = {rvec_l_w.at<double>(0, 0), rvec_l_w.at<double>(1, 0), rvec_l_w.at<double>(2, 0),
                  t_l_w.at<double>(0, 0),    t_l_w.at<double>(1, 0),    t_l_w.at<double>(2, 0)};

    return true;
  }

  tlw_param_ = {0, 0, 0, 0, 0, 0};
  return false;
}

void PTZRayOptimizer::SetUpInitialCameraParams()
{
  intrinsics_param_.clear();
  extrinsics_param_.clear();

  for (size_t i = 0; i < num_cams_; ++i) {
    if (!isCandidate(i))
      continue;
    const auto& param = cameras_[i].ToVector();

    long ic_id = shared_ic_ids_[i];
    if (intrinsics_param_.count(ic_id) == 0) {
      vector<double> intrinsic{param[0], param[1], param[2], param[3], param[10], param[11], param[12], param[13], param[14]};
      intrinsics_param_.insert({ic_id, intrinsic});
    }

    vector<double> extrinsic{param[4], param[5], param[6], param[7], param[8], param[9]};
    extrinsics_param_.insert({i, extrinsic});
  }

  disp_param_ = std::vector<double>(3, 0.0);

  // triangulate rays
  rays_param_.clear();
  for (auto& track_elem : tracks_) {
    int track_id = track_elem.first;
    Track track = track_elem.second;

    cv::Mat ray = Pix2Ray(cameras_, features_, track);
    if (ray.empty())
      continue;

    vector<double> ray_vec = {ray.at<double>(0, 0), ray.at<double>(1, 0), ray.at<double>(2, 0)};
    rays_param_.insert({track_id, ray_vec});
  }
}

void PTZRayOptimizer::ObtainRefinedCameraParams(vector<Camera>& cameras, std::vector<std::vector<Ray>>& rays)
{
  for (size_t i = 0; i < num_cams_; ++i) {
    if (!isCandidate(i))
      continue;

    vector<double> param(15);
    long ic_id = shared_ic_ids_[i];

    double displacement;
    switch (type_) {
      case PTZRayFxfyDist:
        param[0] = intrinsics_param_.at(ic_id)[0];
        param[1] = intrinsics_param_.at(ic_id)[1];
        param[2] = intrinsics_param_.at(ic_id)[2];
        param[3] = intrinsics_param_.at(ic_id)[3];
        param[4] = extrinsics_param_.at(i)[0];
        param[5] = extrinsics_param_.at(i)[1];
        param[6] = extrinsics_param_.at(i)[2];
        param[7] = extrinsics_param_.at(i)[3];
        param[8] = extrinsics_param_.at(i)[4];
        displacement = disp_param_[0] + disp_param_[1] * param[0] + disp_param_[2] * param[0] * param[0];
        param[9] = extrinsics_param_.at(i)[5] + displacement;
        param[10] = intrinsics_param_.at(ic_id)[4];
        param[11] = intrinsics_param_.at(ic_id)[5];
        param[12] = intrinsics_param_.at(ic_id)[6];
        param[13] = intrinsics_param_.at(ic_id)[7];
        param[14] = intrinsics_param_.at(ic_id)[8];
        break;

      case PTZRay:
      case PTZRayDist:
      case PTZRayDistDisp:
        param[0] = intrinsics_param_.at(ic_id)[0];
        param[1] = intrinsics_param_.at(ic_id)[0];
        param[2] = intrinsics_param_.at(ic_id)[2];
        param[3] = intrinsics_param_.at(ic_id)[3];
        param[4] = extrinsics_param_.at(i)[0];
        param[5] = extrinsics_param_.at(i)[1];
        param[6] = extrinsics_param_.at(i)[2];
        param[7] = extrinsics_param_.at(i)[3];
        param[8] = extrinsics_param_.at(i)[4];
        displacement = disp_param_[0] + disp_param_[1] * param[0] + disp_param_[2] * param[0] * param[0];
        param[9] = extrinsics_param_.at(i)[5] + displacement;
        param[10] = intrinsics_param_.at(ic_id)[4];
        param[11] = intrinsics_param_.at(ic_id)[5];
        param[12] = intrinsics_param_.at(ic_id)[6];
        param[13] = intrinsics_param_.at(ic_id)[7];
        param[14] = intrinsics_param_.at(ic_id)[8];
        break;

      default:
        break;
    }

    cameras[i].FromVector(param);
  }

  // Transform all cameras from local to world
  // T_i_w = T_i_l * T_l_w
  cv::Mat R_l_w, t_l_w;
  T_l_w(tlw_param_.data(), R_l_w, t_l_w);

  for (size_t i = 0; i < num_cams_; ++i) {
    if (!isCandidate(i))
      continue;
    cameras[i].t() = cameras[i].R() * t_l_w + cameras[i].t();
    cameras[i].R() = cameras[i].R() * R_l_w;
  }

  // obtain rays
  rays.clear();
  rays.resize(num_cams_);
  for (auto& track_elem : tracks_) {
    int track_id = track_elem.first;
    Track track = track_elem.second;

    cv::Mat R_w_l, t_w_l;
    t_w_l = -R_l_w.t() * t_l_w;
    R_w_l = R_l_w.t();

    vector<double> xyz = rays_param_.at(track_id);
    cv::Mat ray_l = (cv::Mat_<double>(3, 1) << xyz[0], xyz[1], xyz[2]);
    cv::Mat ray_w = R_w_l * ray_l + t_w_l;

    for (auto& iter : track) {
      int image_id = iter.first;
      int feature_id = iter.second;

      cv::Point2f uv = features_[image_id].keypoints[feature_id].pt;

      rays[image_id].emplace_back(track_id, ray_w, uv);
    }
  }
}

cv::Mat PTZRayOptimizer::Pix2Ray(const vector<Camera>& cameras, const vector<ImageFeatures>& features, const Track& track) const
{
  if (cameras.empty() || cameras.size() != features.size()) {
    return cv::Mat();
  }

  size_t track_size = 0;
  cv::Mat ray = cv::Mat_<double>::zeros(3, 1);
  for (auto& track_elem : track) {
    int image_id = track_elem.first;
    int feature_id = track_elem.second;

    if (!isCandidate(image_id))
      continue;

    cv::Point2f pt = features[image_id].keypoints[feature_id].pt;
    cv::Mat uv = (cv::Mat_<double>(3, 1) << pt.x, pt.y, 1);

    cv::Mat ray_temp = cameras[image_id].R().inv() * cameras[image_id].K().inv() * uv;
    ray_temp /= cv::norm(ray_temp);

    ray += ray_temp;
    ++track_size;
  }

  ray /= track_size;
  ray /= cv::norm(ray);

  return ray;
}

void PTZRayOptimizer::AddConstraints2d2d()
{
  for (auto& track_elem : tracks_) {
    int track_id = track_elem.first;
    Track track = track_elem.second;

    double weight = static_cast<double>(track.size());
    ceres::LossFunction* loss_f = new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP);
    // double delta = 2.0;
    // ceres::LossFunction* loss_f = new ceres::HuberLoss(delta);

    for (auto& iter : track) {
      int image_id = iter.first;
      int feature_id = iter.second;
      long ic_id = shared_ic_ids_[image_id];

      if (!isCandidate(image_id))
        continue;

      cv::Point2f uv = features_[image_id].keypoints[feature_id].pt;

      ceres::CostFunction* cost_function = nullptr;
      switch (type_) {
        case PTZRay:
          cost_function = PTZRayFactor::Create(uv);
          problem_.AddResidualBlock(cost_function, loss_f, intrinsics_param_.at(ic_id).data(), extrinsics_param_.at(image_id).data(),
                                    rays_param_.at(track_id).data());
          break;

        case PTZRayDist:
          cost_function = PTZRayDistFactor::Create(uv);
          problem_.AddResidualBlock(cost_function, loss_f, intrinsics_param_.at(ic_id).data(), extrinsics_param_.at(image_id).data(),
                                    rays_param_.at(track_id).data());
          break;

        case PTZRayFxfyDist:
          cost_function = PTZRayFxfyDistFactor::Create(uv);
          problem_.AddResidualBlock(cost_function, loss_f, intrinsics_param_.at(ic_id).data(), extrinsics_param_.at(image_id).data(),
                                    rays_param_.at(track_id).data());
          break;

        case PTZRayDistDisp:
          cost_function = PTZRayDistDispFactor::Create(uv);
          problem_.AddResidualBlock(cost_function, loss_f, intrinsics_param_.at(ic_id).data(), disp_param_.data(),
                                    extrinsics_param_.at(image_id).data(), rays_param_.at(track_id).data());
          break;

        default:
          break;
      }
    }
  }

  // fix some parameters
  for (size_t i = 0; i < num_cams_; ++i) {
    if (!isCandidate(i))
      continue;
    long ic_id = shared_ic_ids_[i];

    if (problem_.HasParameterBlock(intrinsics_param_.at(ic_id).data()) &&
        problem_.GetParameterization((intrinsics_param_.at(ic_id).data())) == nullptr) {
      ceres::SubsetParameterization* constant_parameterization_i = nullptr;
      switch (type_) {
        case PTZRay:
          constant_parameterization_i = new ceres::SubsetParameterization(9, {2, 3, 4, 5, 6, 7, 8});  // cx, cy, k1, k2, k3, p1, p2
          problem_.SetParameterization(intrinsics_param_.at(ic_id).data(), constant_parameterization_i);
          break;

        case PTZRayDist:
        case PTZRayFxfyDist:
        case PTZRayDistDisp:
          constant_parameterization_i = new ceres::SubsetParameterization(9, {2, 3, 5, 6, 7, 8});  // cx, cy, k2, k3, p1, p2
          problem_.SetParameterization(intrinsics_param_.at(ic_id).data(), constant_parameterization_i);
          break;

        default:
          break;
      }
    }

    if (problem_.HasParameterBlock(extrinsics_param_.at(i).data()) &&
        problem_.GetParameterization(extrinsics_param_.at(i).data()) == nullptr) {
      auto* constant_parameterization_i = new ceres::SubsetParameterization(6, {3, 4, 5});  // t1, t2, t3
      problem_.SetParameterization(extrinsics_param_.at(i).data(), constant_parameterization_i);
    }
  }
}

void PTZRayOptimizer::AddConstraints2d3d()
{
  double weight = 1;
  ceres::LossFunction* loss_f = new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP);
  // double delta = 2.0;
  // ceres::LossFunction* loss_f = new ceres::HuberLoss(delta);

  for (size_t i = 0; i < num_cams_; ++i) {
    if (!isCandidate(i))
      continue;
    if (pixels_.empty() || pixels_[i].empty())
      continue;

    ceres::CostFunction* cost_function = nullptr;
    long ic_id = shared_ic_ids_[i];

    for (size_t j = 0; j < pixels_[i].size(); ++j) {
      switch (type_) {
        case PTZRay:
        case PTZRayDist:
        case PTZRayFxfyDist:
          cost_function = Reproj2d3dFactor::Create(pixels_[i][j], pts3d_[i][j]);
          problem_.AddResidualBlock(cost_function, loss_f, intrinsics_param_.at(ic_id).data(), extrinsics_param_.at(i).data(),
                                    tlw_param_.data());
          break;

        case PTZRayDistDisp:
          cost_function = Reproj2d3dDispFactor::Create(pixels_[i][j], pts3d_[i][j]);
          problem_.AddResidualBlock(cost_function, loss_f, intrinsics_param_.at(ic_id).data(), disp_param_.data(),
                                    extrinsics_param_.at(i).data(), tlw_param_.data());
          break;

        default:
          break;
      }
    }
  }

  // fix some parameters
  for (size_t i = 0; i < num_cams_; ++i) {
    if (!isCandidate(i))
      continue;
    long ic_id = shared_ic_ids_[i];

    if (problem_.HasParameterBlock(intrinsics_param_.at(ic_id).data()) &&
        problem_.GetParameterization((intrinsics_param_.at(ic_id).data())) == nullptr) {
      ceres::SubsetParameterization* constant_parameterization_i = nullptr;
      switch (type_) {
        case PTZRay:
          constant_parameterization_i = new ceres::SubsetParameterization(9, {2, 3, 4, 5, 6, 7, 8});  // cx, cy, k1, k2, k3, p1, p2
          problem_.SetParameterization(intrinsics_param_.at(ic_id).data(), constant_parameterization_i);
          break;

        case PTZRayDist:
        case PTZRayFxfyDist:
        case PTZRayDistDisp:
          constant_parameterization_i = new ceres::SubsetParameterization(9, {2, 3, 5, 6, 7, 8});  // cx, cy, k2, k3, p1, p2
          problem_.SetParameterization(intrinsics_param_.at(ic_id).data(), constant_parameterization_i);
          break;

        default:
          break;
      }
    }

    if (problem_.HasParameterBlock(extrinsics_param_.at(i).data()) &&
        problem_.GetParameterization(extrinsics_param_.at(i).data()) == nullptr) {
      auto* constant_parameterization_i = new ceres::SubsetParameterization(6, {3, 4, 5});  // t1, t2, t3
      problem_.SetParameterization(extrinsics_param_.at(i).data(), constant_parameterization_i);
    }
  }
}

void PTZRayOptimizer::CalReprojError()
{
  init_reproj_error_all_ = sqrt(2) * sqrt((2 * summary_.initial_cost) / summary_.num_residuals);
  final_reproj_error_all_ = sqrt(2) * sqrt((2 * summary_.final_cost) / summary_.num_residuals);

  CalReprojError2d2d();

  CalReprojError2d3d();
}

void PTZRayOptimizer::CalReprojError2d2d()
{
  double residual_2d2d_sum[2] = {0, 0};
  int residual_2d2d_num = 0;

  for (auto& track_elem : tracks_) {
    int track_id = track_elem.first;
    Track track = track_elem.second;

    for (auto& iter : track) {
      int image_id = iter.first;
      int feature_id = iter.second;
      long ic_id = shared_ic_ids_[image_id];

      if (!isCandidate(image_id))
        continue;

      cv::Point2f uv = features_[image_id].keypoints[feature_id].pt;

      double residual_2d2d[2] = {0, 0};

      switch (type_) {
        case PTZRay: {
          PTZRayFactor factor(uv);
          factor(intrinsics_param_.at(ic_id).data(), extrinsics_param_.at(image_id).data(), rays_param_.at(track_id).data(), residual_2d2d);
          break;
        }

        case PTZRayDist: {
          PTZRayDistFactor factor(uv);
          factor(intrinsics_param_.at(ic_id).data(), extrinsics_param_.at(image_id).data(), rays_param_.at(track_id).data(), residual_2d2d);
          break;
        }

        case PTZRayFxfyDist: {
          PTZRayFxfyDistFactor factor(uv);
          factor(intrinsics_param_.at(ic_id).data(), extrinsics_param_.at(image_id).data(), rays_param_.at(track_id).data(), residual_2d2d);
          break;
        }

        case PTZRayDistDisp: {
          PTZRayDistDispFactor factor(uv);
          factor(intrinsics_param_.at(ic_id).data(), disp_param_.data(), extrinsics_param_.at(image_id).data(),
                 rays_param_.at(track_id).data(), residual_2d2d);
          break;
        }

        default:
          break;
      }

      ++residual_2d2d_num;
      residual_2d2d_sum[0] += residual_2d2d[0] * residual_2d2d[0];
      residual_2d2d_sum[1] += residual_2d2d[1] * residual_2d2d[1];
    }
  }

  final_reproj_error_2d2d_ = sqrt((residual_2d2d_sum[0] + residual_2d2d_sum[1]) / residual_2d2d_num);
}

void PTZRayOptimizer::CalReprojError2d3d()
{
  double residual_2d3d_sum[2] = {0, 0};
  int residual_2d3d_num = 0;

  for (size_t i = 0; i < num_cams_; ++i) {
    if (!isCandidate(i))
      continue;
    if (pixels_.empty() || pixels_[i].empty())
      continue;

    long ic_id = shared_ic_ids_[i];

    for (size_t j = 0; j < pixels_[i].size(); ++j) {
      double residual_2d3d[2] = {0, 0};

      switch (type_) {
        case PTZRay:
        case PTZRayDist:
        case PTZRayFxfyDist: {
          Reproj2d3dFactor factor(pixels_[i][j], pts3d_[i][j]);
          factor(intrinsics_param_.at(ic_id).data(), extrinsics_param_.at(i).data(), tlw_param_.data(), residual_2d3d);
          break;
        }

        case PTZRayDistDisp: {
          Reproj2d3dDispFactor factor(pixels_[i][j], pts3d_[i][j]);
          factor(intrinsics_param_.at(ic_id).data(), disp_param_.data(), extrinsics_param_.at(i).data(), tlw_param_.data(), residual_2d3d);
          break;
        }

        default:
          break;
      }

      ++residual_2d3d_num;
      residual_2d3d_sum[0] += residual_2d3d[0] * residual_2d3d[0];
      residual_2d3d_sum[1] += residual_2d3d[1] * residual_2d3d[1];
    }
  }

  final_reproj_error_2d3d_ = sqrt((residual_2d3d_sum[0] + residual_2d3d_sum[1]) / residual_2d3d_num);
}

}  // namespace ptzcalib
