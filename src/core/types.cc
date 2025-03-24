/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 19:43:22
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-03-12 14:42:42
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#include "types.h"

using namespace std;

namespace ptzcalib {

Camera::Camera()
{
  K_ = cv::Mat_<double>::eye(3, 3);
  R_ = cv::Mat_<double>::eye(3, 3);
  t_ = cv::Mat_<double>::zeros(3, 1);
  dist_ = cv::Mat_<double>::zeros(5, 1);
}

Camera::Camera(const cv::Mat& K, const cv::Mat& R, const cv::Mat& t, const cv::Mat& dist)
{
  K_ = K.clone();
  R_ = R.clone();
  t_ = t.clone();
  dist_ = dist.clone();
}

vector<double> Camera::ToVector() const
{
  vector<double> v(15);
  v[0] = K_.at<double>(0, 0);  // fx
  v[1] = K_.at<double>(1, 1);  // fy
  v[2] = K_.at<double>(0, 2);  // cx
  v[3] = K_.at<double>(1, 2);  // cy

  cv::Mat rvec;
  cv::Rodrigues(R_, rvec);
  v[4] = rvec.at<double>(0, 0);
  v[5] = rvec.at<double>(1, 0);
  v[6] = rvec.at<double>(2, 0);  // rvec

  v[7] = t_.at<double>(0, 0);
  v[8] = t_.at<double>(1, 0);
  v[9] = t_.at<double>(2, 0);  // t

  v[10] = dist_.at<double>(0, 0);
  v[11] = dist_.at<double>(1, 0);
  v[12] = dist_.at<double>(2, 0);
  v[13] = dist_.at<double>(3, 0);
  v[14] = dist_.at<double>(4, 0);  // dist

  return v;
}

void Camera::FromVector(const vector<double>& v)
{
  if (v.size() != 15) {
    throw invalid_argument("Expected camera vector size: 15, actual size :" + to_string(v.size()));
  }

  K_ = (cv::Mat_<double>(3, 3) << v[0], 0, v[2], 0, v[1], v[3], 0, 0, 1);

  cv::Mat rvec = (cv::Mat_<double>(3, 1) << v[4], v[5], v[6]);
  cv::Rodrigues(rvec, R_);

  t_ = (cv::Mat_<double>(3, 1) << v[7], v[8], v[9]);

  dist_ = (cv::Mat_<double>(5, 1) << v[10], v[11], v[12], v[13], v[14]);
}

cv::Mat pt3d_to_pix(const Camera& camera, const cv::Mat& pt3d)
{
  cv::Mat pt_cam = camera.R() * pt3d + camera.t();
  double near_plane = 1.0;  // 1m
  if (pt_cam.at<double>(2, 0) < near_plane)
    return cv::Mat();

  pt_cam /= pt_cam.at<double>(2, 0);
  cv::Mat uv = camera.K() * pt_cam;
  return uv;
}

}  // namespace ptzcalib
