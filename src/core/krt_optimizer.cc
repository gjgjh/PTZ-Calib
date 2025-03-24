/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 20:06:16
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-03-12 17:17:51
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#include "krt_optimizer.h"

#include <opencv2/calib3d.hpp>

#include "utils/logging.h"

using namespace std;

namespace ptzcalib {

/////////////////////////////////////// Factor2d2d ////////////////////////////////////////////////

bool Factor2d2d::operator()(const double* const camera, double* residual) const
{
  const int N = 15;
  vector<double> param(camera, camera + N);
  param[1] = param[0];  // fx = fy

  Camera cam2;
  cam2.FromVector(param);

  cv::Mat pt1 = (cv::Mat_<double>(3, 1) << uv1_.x, uv1_.y, 1);
  cv::Mat ray1 = cam1_.R().inv() * cam1_.K().inv() * pt1;
  ray1 /= cv::norm(ray1);

  // x2 = H21*x1 = K2*R2*R1^(-1)*K1^(-1)*x1
  cv::Mat uv2_predict = cam2.K() * cam2.R() * ray1;
  uv2_predict /= uv2_predict.at<double>(2, 0);

  residual[0] = uv2_.x - uv2_predict.at<double>(0, 0);
  residual[1] = uv2_.y - uv2_predict.at<double>(1, 0);

  return true;
}

ceres::CostFunction* Factor2d2d::Create(const Camera& cam1, const cv::Point2f& uv1, const cv::Point2f& uv2)
{
  return (new ceres::NumericDiffCostFunction<Factor2d2d, ceres::CENTRAL, 2, 15>(new Factor2d2d(cam1, uv1, uv2)));
}

/////////////////////////////////////// Factor2d2dFxfy ////////////////////////////////////////////////

bool Factor2d2dFxfy::operator()(const double* const camera, double* residual) const
{
  const int N = 15;
  vector<double> param(camera, camera + N);

  Camera cam2;
  cam2.FromVector(param);

  cv::Mat pt1 = (cv::Mat_<double>(3, 1) << uv1_.x, uv1_.y, 1);
  cv::Mat ray1 = cam1_.R().inv() * cam1_.K().inv() * pt1;

  // x2 = H21*x1 = K2*R2*R1^(-1)*K1^(-1)*x1
  cv::Mat uv2_predict = cam2.K() * cam2.R() * ray1;
  uv2_predict /= uv2_predict.at<double>(2, 0);

  residual[0] = uv2_.x - uv2_predict.at<double>(0, 0);
  residual[1] = uv2_.y - uv2_predict.at<double>(1, 0);

  return true;
}

ceres::CostFunction* Factor2d2dFxfy::Create(const Camera& cam1, const cv::Point2f& uv1, const cv::Point2f& uv2)
{
  return (new ceres::NumericDiffCostFunction<Factor2d2dFxfy, ceres::CENTRAL, 2, 15>(new Factor2d2dFxfy(cam1, uv1, uv2)));
}

/////////////////////////////////////// Factor2d2dDist ////////////////////////////////////////////////

bool Factor2d2dDist::operator()(const double* const camera, double* residual) const
{
  const int N = 15;
  vector<double> param(camera, camera + N);
  param[1] = param[0];  // fx = fy

  Camera cam2;
  cam2.FromVector(param);

  vector<cv::Point2f> uv1_origin{uv1_};
  vector<cv::Point2f> uv1_undistort;
  cv::undistortPoints(uv1_origin, uv1_undistort, cam1_.K(), cam1_.dist(), cv::noArray(), cam1_.K());
  cv::Mat pt1 = (cv::Mat_<double>(3, 1) << uv1_undistort[0].x, uv1_undistort[0].y, 1);

  // avoid near-boarder pixels
  double width1 = cam1_.K().at<double>(0, 2) * 2;
  double height1 = cam1_.K().at<double>(1, 2) * 2;
  if (uv1_undistort[0].x < 0 || uv1_undistort[0].x >= width1 || uv1_undistort[0].y < 0 || uv1_undistort[0].y >= height1) {
    residual[0] = 0;
    residual[1] = 0;
    return true;
  }

  cv::Mat ray1 = cam1_.R().inv() * cam1_.K().inv() * pt1;
  ray1 /= cv::norm(ray1);

  // x2 = K2*R2*X1
  cv::Mat pt3d_2 = cam2.R() * ray1;
  pt3d_2 /= pt3d_2.at<double>(2, 0);
  double x = pt3d_2.at<double>(0, 0), y = pt3d_2.at<double>(1, 0);

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

  double fx = param[0], fy = param[1], cx = param[2], cy = param[3];
  double x_proj = fx * x_distorted + cx;
  double y_proj = fy * y_distorted + cy;

  residual[0] = uv2_.x - x_proj;
  residual[1] = uv2_.y - y_proj;

  return true;
}

ceres::CostFunction* Factor2d2dDist::Create(const Camera& cam1, const cv::Point2f& uv1, const cv::Point2f& uv2)
{
  return (new ceres::NumericDiffCostFunction<Factor2d2dDist, ceres::CENTRAL, 2, 15>(new Factor2d2dDist(cam1, uv1, uv2)));
}

/////////////////////////////////////// Factor2d2dFxfyDist ////////////////////////////////////////////////

bool Factor2d2dFxfyDist::operator()(const double* const camera, double* residual) const
{
  const int N = 15;
  vector<double> param(camera, camera + N);

  Camera cam2;
  cam2.FromVector(param);

  vector<cv::Point2f> uv1_origin{uv1_};
  vector<cv::Point2f> uv1_undistort;
  cv::undistortPoints(uv1_origin, uv1_undistort, cam1_.K(), cam1_.dist(), cv::noArray(), cam1_.K());
  cv::Mat pt1 = (cv::Mat_<double>(3, 1) << uv1_undistort[0].x, uv1_undistort[0].y, 1);

  // avoid near-boarder pixels
  double width1 = cam1_.K().at<double>(0, 2) * 2;
  double height1 = cam1_.K().at<double>(1, 2) * 2;
  if (uv1_undistort[0].x < 0 || uv1_undistort[0].x >= width1 || uv1_undistort[0].y < 0 || uv1_undistort[0].y >= height1) {
    residual[0] = 0;
    residual[1] = 0;
    return true;
  }

  cv::Mat ray1 = cam1_.R().inv() * cam1_.K().inv() * pt1;
  ray1 /= cv::norm(ray1);

  // x2 = K2*R2*X1
  cv::Mat pt3d_2 = cam2.R() * ray1;
  pt3d_2 /= pt3d_2.at<double>(2, 0);
  double x = pt3d_2.at<double>(0, 0), y = pt3d_2.at<double>(1, 0);

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

  double fx = param[0], fy = param[1], cx = param[2], cy = param[3];
  double x_proj = fx * x_distorted + cx;
  double y_proj = fy * y_distorted + cy;

  residual[0] = uv2_.x - x_proj;
  residual[1] = uv2_.y - y_proj;

  return true;
}

ceres::CostFunction* Factor2d2dFxfyDist::Create(const Camera& cam1, const cv::Point2f& uv1, const cv::Point2f& uv2)
{
  return (new ceres::NumericDiffCostFunction<Factor2d2dFxfyDist, ceres::CENTRAL, 2, 15>(new Factor2d2dFxfyDist(cam1, uv1, uv2)));
}

/////////////////////////////////////// Factor2d3dDist ////////////////////////////////////////////////

bool Factor2d3dDist::operator()(const double* const camera, double* residual) const
{
  const int N = 15;
  vector<double> param(camera, camera + N);
  param[1] = param[0];  // fx = fy

  Camera cam;
  cam.FromVector(param);

  vector<cv::Point2d> pts2d;
  vector<cv::Point3d> pts3d{pt3d_};
  cv::projectPoints(pts3d, cam.rvec(), cam.t(), cam.K(), cam.dist(), pts2d);

  residual[0] = pt2d_.x - pts2d[0].x;
  residual[1] = pt2d_.y - pts2d[0].y;

  return true;
}

ceres::CostFunction* Factor2d3dDist::Create(const cv::Point2f& pt2d, const cv::Point3d& pt3d)
{
  return (new ceres::NumericDiffCostFunction<Factor2d3dDist, ceres::CENTRAL, 2, 15>(new Factor2d3dDist(pt2d, pt3d)));
}

/////////////////////////////////////// Factor2d3dFxfyDist ////////////////////////////////////////////////

bool Factor2d3dFxfyDist::operator()(const double* const camera, double* residual) const
{
  const int N = 15;
  vector<double> param(camera, camera + N);

  Camera cam;
  cam.FromVector(param);

  vector<cv::Point2d> pts2d;
  vector<cv::Point3d> pts3d{pt3d_};
  cv::projectPoints(pts3d, cam.rvec(), cam.t(), cam.K(), cam.dist(), pts2d);

  residual[0] = pt2d_.x - pts2d[0].x;
  residual[1] = pt2d_.y - pts2d[0].y;

  return true;
}

ceres::CostFunction* Factor2d3dFxfyDist::Create(const cv::Point2f& pt2d, const cv::Point3d& pt3d)
{
  return (new ceres::NumericDiffCostFunction<Factor2d3dFxfyDist, ceres::CENTRAL, 2, 15>(new Factor2d3dFxfyDist(pt2d, pt3d)));
}

/////////////////////////////////////// KRTOptimizer ////////////////////////////////////////////////

KRTOptimizer::KRTOptimizer(int max_iter, double max_reproj_error, FACTOR_TYPE factor_type)
    : max_iter_(max_iter), max_reproj_error_(max_reproj_error), factor_type_(factor_type)
{
}

void KRTOptimizer::SetInitParams(const cv::Mat& K, const cv::Mat& R, const cv::Mat& t, const cv::Mat& dist)
{
  cam_curr_world_.K() = K.clone();
  cam_curr_world_.dist() = dist.clone();
  cam_curr_world_.R() = R.clone();
  cam_curr_world_.t() = t.clone();
}

void KRTOptimizer::Add2d2dConstraints(const Camera& cam_ref, const std::vector<cv::KeyPoint>& kpts_ref,
                                      const std::vector<cv::KeyPoint>& kpts_curr, const std::vector<cv::DMatch>& matches)
{
  // select reference frame as local coordinate, transform all poses from world to local
  R_local_world_ = cam_ref.R();
  t_local_world_ = cam_ref.t();

  Camera cam_ref_local;
  cam_ref_local.K() = cam_ref.K();
  cam_ref_local.dist() = cam_ref.dist();
  cam_ref_local.R() = cv::Mat_<double>::eye(3, 3);
  cam_ref_local.t() = cv::Mat_<double>::zeros(3, 1);

  // T_curr_local = T_curr_world * T_local_world^{-1}
  cam_curr_local_.K() = cam_curr_world_.K();
  cam_curr_local_.dist() = cam_curr_world_.dist();
  cam_curr_local_.R() = cam_curr_world_.R() * R_local_world_.inv();
  cam_curr_local_.t() = -cam_curr_world_.R() * R_local_world_.inv() * t_local_world_ + cam_curr_world_.t();

  cam_curr_local_param_ = cam_curr_local_.ToVector();

  for (auto& match : matches) {
    cv::Point2f uv1 = kpts_ref[match.queryIdx].pt;
    cv::Point2f uv2 = kpts_curr[match.trainIdx].pt;

    ceres::CostFunction* cost_function = nullptr;
    switch (factor_type_) {
      case F:
        cost_function = Factor2d2d::Create(cam_ref_local, uv1, uv2);
        problem_.AddResidualBlock(cost_function, nullptr /* squared loss */, cam_curr_local_param_.data());
        break;

      case FDist:
        cost_function = Factor2d2dDist::Create(cam_ref_local, uv1, uv2);
        problem_.AddResidualBlock(cost_function, nullptr /* squared loss */, cam_curr_local_param_.data());
        break;

      case Fxfy:
        cost_function = Factor2d2dFxfy::Create(cam_ref_local, uv1, uv2);
        problem_.AddResidualBlock(cost_function, nullptr /* squared loss */, cam_curr_local_param_.data());
        break;

      case FxfyDist:
        cost_function = Factor2d2dFxfyDist::Create(cam_ref_local, uv1, uv2);
        problem_.AddResidualBlock(cost_function, nullptr /* squared loss */, cam_curr_local_param_.data());
        break;

      default:
        break;
    }
  }

  // fix some parameters
  if (problem_.HasParameterBlock(cam_curr_local_param_.data()) && problem_.GetParameterization(cam_curr_local_param_.data()) == nullptr) {
    ceres::SubsetParameterization* constant_parameterization = nullptr;
    switch (factor_type_) {
      case F:
        constant_parameterization =
            new ceres::SubsetParameterization(15, {1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14});  // fy, cx, cy, t[0:2], dist[0:4]
        problem_.SetParameterization(cam_curr_local_param_.data(), constant_parameterization);
        break;

      case Fxfy:
        constant_parameterization =
            new ceres::SubsetParameterization(15, {2, 3, 7, 8, 9, 10, 11, 12, 13, 14});  // cx, cy, t[0:2], dist[0:4]
        problem_.SetParameterization(cam_curr_local_param_.data(), constant_parameterization);
        break;

      case FDist:
        constant_parameterization =
            new ceres::SubsetParameterization(15, {1, 2, 3, 7, 8, 9, 11, 12, 13, 14});  // fy, cx, cy, t[0:2], dist[1:4]
        problem_.SetParameterization(cam_curr_local_param_.data(), constant_parameterization);
        break;

      case FxfyDist:
        constant_parameterization = new ceres::SubsetParameterization(15, {2, 3, 7, 8, 9, 11, 12, 13, 14});  // cx, cy, t[0:2], dist[1:4]
        problem_.SetParameterization(cam_curr_local_param_.data(), constant_parameterization);
        break;

      default:
        break;
    }
  }
}

void KRTOptimizer::Add2d3dConstraints(const std::vector<cv::Point2f>& pts2d, const std::vector<cv::Point3d>& pts3d)
{
  if (pts2d.size() != pts3d.size() || pts2d.empty())
    return;
  const size_t num_points = pts2d.size();

  // convert pts3d to local coordinate
  std::vector<cv::Point3d> pts3d_local;
  for (size_t i = 0; i < num_points; ++i) {
    cv::Mat pt3d_world = (cv::Mat_<double>(3, 1) << pts3d[i].x, pts3d[i].y, pts3d[i].z);
    cv::Mat pt3d_local = R_local_world_ * pt3d_world + t_local_world_;
    pts3d_local.emplace_back(pt3d_local.at<double>(0, 0), pt3d_local.at<double>(1, 0), pt3d_local.at<double>(2, 0));
  }

  for (size_t i = 0; i < num_points; ++i) {
    ceres::CostFunction* cost_function = nullptr;
    switch (factor_type_) {
      case F:
      case FDist:
        cost_function = Factor2d3dDist::Create(pts2d[i], pts3d_local[i]);
        problem_.AddResidualBlock(cost_function, nullptr /* squared loss */, cam_curr_local_param_.data());
        break;

      case Fxfy:
      case FxfyDist:
        cost_function = Factor2d3dFxfyDist::Create(pts2d[i], pts3d_local[i]);
        problem_.AddResidualBlock(cost_function, nullptr /* squared loss */, cam_curr_local_param_.data());
        break;

      default:
        break;
    }
  }
}

bool KRTOptimizer::Solve(cv::Mat& K, cv::Mat& R, cv::Mat& t, cv::Mat& dist)
{
  ceres::Solver::Options options;
  options.max_num_iterations = max_iter_;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 32;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem_, &summary);

  num_iter_ = summary.num_successful_steps;

  if (!CheckResults(summary)) {
    return false;
  }

  ObtainRefinedCameraParams(K, R, t, dist);
  return true;
}

double KRTOptimizer::Cal2d2dReprojError(const Camera& cam_ref, const std::vector<cv::KeyPoint>& kpts_ref,
                                        const std::vector<cv::KeyPoint>& kpts_curr, const std::vector<cv::DMatch>& matches)
{
  Camera cam_ref_local;
  cam_ref_local.K() = cam_ref.K();
  cam_ref_local.dist() = cam_ref.dist();
  cam_ref_local.R() = cv::Mat_<double>::eye(3, 3);
  cam_ref_local.t() = cv::Mat_<double>::zeros(3, 1);

  double residuals[2] = {0, 0};

  for (auto& match : matches) {
    cv::Point2f uv1 = kpts_ref[match.queryIdx].pt;
    cv::Point2f uv2 = kpts_curr[match.trainIdx].pt;

    double res[2] = {0, 0};
    switch (factor_type_) {
      case F: {
        Factor2d2d factor(cam_ref_local, uv1, uv2);
        factor(cam_curr_local_param_.data(), res);
        break;
      }
      case Fxfy: {
        Factor2d2dFxfy factor(cam_ref_local, uv1, uv2);
        factor(cam_curr_local_param_.data(), res);
        break;
      }
      case FDist: {
        Factor2d2dDist factor(cam_ref_local, uv1, uv2);
        factor(cam_curr_local_param_.data(), res);
        break;
      }
      case FxfyDist: {
        Factor2d2dFxfyDist factor(cam_ref_local, uv1, uv2);
        factor(cam_curr_local_param_.data(), res);
        break;
      }
      default:
        break;
    }

    residuals[0] += res[0] * res[0];
    residuals[1] += res[1] * res[1];
  }

  double num_observation = static_cast<double>(matches.size());
  double reproj_error = sqrt((residuals[0] + residuals[1]) / num_observation);

  return reproj_error;
}

double KRTOptimizer::Cal2d3dReprojError(const std::vector<cv::Point2f>& pts2d, const std::vector<cv::Point3d>& pts3d)
{
  if (pts2d.size() != pts3d.size() || pts2d.empty())
    return -1;
  const size_t num_points = pts2d.size();

  // convert pts3d to local coordinate
  std::vector<cv::Point3d> pts3d_local;
  for (size_t i = 0; i < num_points; ++i) {
    cv::Mat pt3d_world = (cv::Mat_<double>(3, 1) << pts3d[i].x, pts3d[i].y, pts3d[i].z);
    cv::Mat pt3d_local = R_local_world_ * pt3d_world + t_local_world_;
    pts3d_local.emplace_back(pt3d_local.at<double>(0, 0), pt3d_local.at<double>(1, 0), pt3d_local.at<double>(2, 0));
  }

  double residuals[2] = {0, 0};

  for (size_t i = 0; i < num_points; ++i) {
    double res[2] = {0, 0};
    switch (factor_type_) {
      case F:
      case FDist: {
        Factor2d3dDist factor(pts2d[i], pts3d_local[i]);
        factor(cam_curr_local_param_.data(), res);
        break;
      }
      case Fxfy:
      case FxfyDist: {
        Factor2d3dFxfyDist factor(pts2d[i], pts3d_local[i]);
        factor(cam_curr_local_param_.data(), res);
        break;
      }
      default:
        break;
    }

    residuals[0] += res[0] * res[0];
    residuals[1] += res[1] * res[1];
  }

  double num_observation = static_cast<double>(num_points);
  double reproj_error = sqrt((residuals[0] + residuals[1]) / num_observation);

  return reproj_error;
}

void KRTOptimizer::SetFixedFocal() { set_fixed_focal_ = true; }

bool KRTOptimizer::CheckResults(const ceres::Solver::Summary& summary)
{
  double init_reproj_error = sqrt(2) * sqrt((2 * summary.initial_cost) / summary.num_residuals);
  double final_reproj_error = sqrt(2) * sqrt((2 * summary.final_cost) / summary.num_residuals);

  PLOGD << summary.BriefReport();
  PLOGD << "Init reprojection error: " << init_reproj_error << ", final reprojection error: " << final_reproj_error;

  // check convergence
  if (summary.termination_type != ceres::TerminationType::CONVERGENCE)
    return false;

  // check reprojection error
  if (final_reproj_error >= max_reproj_error_)
    return false;

  // check focal length and fov
  Camera cam;
  cam.FromVector(cam_curr_local_param_);
  double fx = cam.K().at<double>(0, 0), fy = cam.K().at<double>(1, 1);
  double cx = cam.K().at<double>(0, 2), cy = cam.K().at<double>(1, 2);
  double fov_x = atan(cx / fx) * 2 * 180 / M_PI;
  double fov_y = atan(cy / fy) * 2 * 180 / M_PI;
  if (fov_x < 0 || fov_x > 170 || fov_y < 0 || fov_y > 170) {
    PLOGD << "FOV invalid! fov_x: " << fov_x << ", fov_y: " << fov_y;
    return false;
  }

  return true;
}

void KRTOptimizer::ObtainRefinedCameraParams(cv::Mat& K, cv::Mat& R, cv::Mat& t, cv::Mat& dist)
{
  vector<double> param(15);

  switch (factor_type_) {
    case F:
    case FDist:
      param = cam_curr_local_param_;
      param[1] = param[0];  // fx = fy
      break;

    case Fxfy:
    case FxfyDist:
      param = cam_curr_local_param_;
      break;

    default:
      break;
  }

  cam_curr_local_.FromVector(param);

  // transform from local to world: T_curr_world = T_curr_local * T_local_world
  cam_curr_world_.K() = cam_curr_local_.K();
  cam_curr_world_.dist() = cam_curr_local_.dist();
  cam_curr_world_.R() = cam_curr_local_.R() * R_local_world_;
  cam_curr_world_.t() = cam_curr_local_.R() * t_local_world_ + cam_curr_local_.t();

  K = cam_curr_world_.K();
  dist = cam_curr_world_.dist();
  R = cam_curr_world_.R();
  t = cam_curr_world_.t();
}

}  // namespace ptzcalib
