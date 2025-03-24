/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 16:33:31
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-03-12 14:42:01
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#ifndef PTZ_CALIB_SRC_CORE_TYPES_H
#define PTZ_CALIB_SRC_CORE_TYPES_H

#include <opencv2/calib3d.hpp>

namespace ptzcalib {

struct ImageFeatures {
  long img_idx;
  cv::Size img_size;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};

struct MatchesInfo {
  long src_img_idx;
  long dst_img_idx;  //!< Images indices
  std::vector<cv::DMatch> matches;
  std::vector<uchar> inliers_mask;  //!< Geometrically consistent matches mask
  int num_inliers;                  //!< Number of geometrically consistent matches
  cv::Mat H;                        //!< Estimated transformation
  double confidence;                //!< Confidence two images are from the same panorama
};

struct Ray {
  int id_;
  cv::Point3d pt3d_;
  cv::Point2f uv_;

  Ray(int id, const cv::Mat& pt3d, const cv::Point2f& uv) : id_(id), uv_(uv)
  {
    pt3d_.x = pt3d.at<double>(0, 0);
    pt3d_.y = pt3d.at<double>(0, 1);
    pt3d_.z = pt3d.at<double>(0, 2);
  }
};

class Camera {
 public:
  Camera();
  Camera(const cv::Mat& K, const cv::Mat& R, const cv::Mat& t, const cv::Mat& dist);
  ~Camera() = default;

  const cv::Mat& K() const;
  cv::Mat& K();
  const cv::Mat& R() const;
  cv::Mat& R();
  cv::Mat rvec() const;
  const cv::Mat& t() const;
  cv::Mat& t();
  cv::Mat t_wc() const;
  const cv::Mat& dist() const;
  cv::Mat& dist();

  std::vector<double> ToVector() const;
  void FromVector(const std::vector<double>& v);

 private:
  cv::Mat K_;     //!< camera intrinsic matrix 3x3
  cv::Mat R_;     //!< rotation matrix 3x3 (camera to world)
  cv::Mat t_;     //!< camera translation vector 3x1 (camera to world)
  cv::Mat dist_;  //!< camera distortion coefficients 5x1
};

inline const cv::Mat& Camera::K() const { return K_; }

inline cv::Mat& Camera::K() { return K_; }

inline const cv::Mat& Camera::R() const { return R_; }

inline cv::Mat& Camera::R() { return R_; }

inline cv::Mat Camera::rvec() const
{
  cv::Mat rvec;
  cv::Rodrigues(R_, rvec);
  return rvec;
}

inline const cv::Mat& Camera::t() const { return t_; }

inline cv::Mat& Camera::t() { return t_; }

inline cv::Mat Camera::t_wc() const { return -R_.inv() * t_; }

inline const cv::Mat& Camera::dist() const { return dist_; }

inline cv::Mat& Camera::dist() { return dist_; }

cv::Mat pt3d_to_pix(const Camera& camera, const cv::Mat& pt3d);

}  // namespace ptzcalib

#endif  // PTZ_CALIB_SRC_CORE_TYPES_H