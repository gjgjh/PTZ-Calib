/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 17:26:58
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-03-12 18:27:54
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#include "data_io.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <unordered_set>

#include "utils/logging.h"
#include "utils/os_path.h"

using namespace std;

namespace ptzcalib {

void ReadColmapFeatures(const string &filepath, vector<cv::KeyPoint> &kpts, cv::Mat &desc)
{
  kpts.clear();
  desc = cv::Mat();

  try {
    ifstream fin(filepath);
    if (!fin.good())
      return;
    int num_kpts, desc_dim;
    fin >> num_kpts >> desc_dim;
    kpts = vector<cv::KeyPoint>(num_kpts);
    desc = cv::Mat(cv::Size(desc_dim, num_kpts), CV_32F);
    float scale, orientation;

    for (int i = 0; i < num_kpts; ++i) {
      fin >> kpts[i].pt.x >> kpts[i].pt.y >> scale >> orientation;
      for (int j = 0; j < desc_dim; ++j) {
        fin >> desc.at<float>(i, j);
      }
    }
  }
  catch (exception &e) {
    PLOGD << "Cannot read colmap features from: " << filepath;
    kpts.clear();
    desc = cv::Mat();
    return;
  }
}

static bool HasEnding(const string &fullString, const string &ending)
{
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  }
  else {
    return false;
  }
}

void ReadColmapMatches(const string &filepath, vector<vector<cv::DMatch>> &pairs_matches, vector<pair<string, string>> &img_pairs_name)
{
  pairs_matches.clear();
  img_pairs_name.clear();

  vector<cv::DMatch> matches;
  pair<string, string> img_pair;

  try {
    ifstream fin(filepath);
    string line;
    while (getline(fin, line)) {
      if (line.empty()) {
        if (!matches.empty()) {
          pairs_matches.push_back(matches);
          img_pairs_name.push_back(img_pair);

          matches.clear();
          img_pair = {};
        }

        continue;
      }

      string str1, str2;
      istringstream iss(line);
      iss >> str1 >> str2;

      if (HasEnding(str1, ".png") || HasEnding(str1, ".jpg") || HasEnding(str1, ".jpeg")) {
        img_pair = {str1, str2};
      }
      else {
        int i = stoi(str1), j = stoi(str2);
        matches.emplace_back(i, j, 0);
      }
    }

    fin.close();
  }
  catch (exception &e) {
    PLOGD << "Cannot read colmap matches from: " << filepath;
  }
}

bool SaveToJson(const std::vector<Camera> &cameras, const std::vector<std::string> &names,
                const std::vector<std::vector<cv::Point2f>> &pixels_gt, const std::vector<std::vector<cv::Point3d>> &pts3d_gt,
                const std::string &filepath)
{
  nlohmann::ordered_json j_all;
  for (size_t i = 0; i < cameras.size(); ++i) {
    nlohmann::ordered_json j;

    string rootname, ext;
    splitext(names[i], &rootname, &ext);

    j["name"] = rootname;
    cv::Mat t_wc = cameras[i].t_wc();
    vector<double> t_wc_vec{t_wc.at<double>(0, 0), t_wc.at<double>(1, 0), t_wc.at<double>(2, 0)};
    j["pos"] = t_wc_vec;
    int width = static_cast<int>(2 * cameras[i].K().at<double>(0, 2));
    int height = static_cast<int>(2 * cameras[i].K().at<double>(1, 2));
    j["res"] = {width, height};
    vector<double> vec;
    vec.assign(cameras[i].K().begin<double>(), cameras[i].K().end<double>());
    j["K"] = vec;
    vec.assign(cameras[i].R().begin<double>(), cameras[i].R().end<double>());
    j["R"] = vec;
    vec.assign(cameras[i].t().begin<double>(), cameras[i].t().end<double>());
    j["t"] = vec;

    vec.assign(cameras[i].dist().begin<double>(), cameras[i].dist().end<double>());
    j["dist"] = vec;
    if (vec[0] < 1e-5)
      j["distType"] = "";
    else
      j["distType"] = "k1";

    vector<vector<float>> pix;
    vector<vector<double>> pos;
    for (size_t k = 0; k < pixels_gt[i].size(); ++k) {
      pix.push_back({pixels_gt[i][k].x / width, pixels_gt[i][k].y / height});
      pos.push_back({pts3d_gt[i][k].x, pts3d_gt[i][k].y, pts3d_gt[i][k].z});
    }
    j["marker"] = {{"pix", pix}, {"pos", pos}};

    j["version"] = "2.0";

    j_all["cameras"][rootname] = j;
  }

  ofstream fout(filepath, ios_base::out);
  fout << j_all.dump(4) << endl;
  fout.close();

  return true;
}

static nlohmann::json ReadJsonFile(const string &filepath) noexcept
{
  if (filepath.empty()) {
    return nlohmann::json::object();
  }

  try {
    ifstream in(filepath, ios::in);
    if (!in.is_open() || in.fail()) {
      return nlohmann::json::object();
    }

    nlohmann::json json;
    in >> json;
    return json;
  }
  catch (exception &e) {
    return nlohmann::json::object();
  }
}

bool ReadFromJson(const std::string &filepath, std::vector<Camera> &cameras, std::vector<std::string> &names,
                  std::vector<std::vector<cv::Point2f>> &pixels, std::vector<std::vector<cv::Point3d>> &pts3d, std::vector<cv::Size> &sizes)
{
  cameras.clear();
  names.clear();
  pixels.clear();
  pts3d.clear();
  sizes.clear();

  auto j = ReadJsonFile(filepath);
  if (j.empty()) {
    PLOGE << "JSON File not exists or cannot be opened: " + filepath;
    return false;
  }

  try {
    auto j_cameras = j["cameras"];
    for (auto &el : j_cameras.items()) {
      string name = el.key();
      std::vector<cv::Point2f> pixs;
      std::vector<cv::Point3d> pts;
      Camera cam;
      cv::Size size;

      auto value = el.value();
      std::vector<double> vec;
      value["K"].get_to(vec);
      memcpy(cam.K().data, vec.data(), vec.size() * sizeof(double));
      value["R"].get_to(vec);
      memcpy(cam.R().data, vec.data(), vec.size() * sizeof(double));
      value["t"].get_to(vec);
      memcpy(cam.t().data, vec.data(), vec.size() * sizeof(double));
      value["dist"].get_to(vec);
      memcpy(cam.dist().data, vec.data(), vec.size() * sizeof(double));

      int width = value["res"][0];
      int height = value["res"][1];
      size.width = width;
      size.height = height;

      vector<vector<double>> pixels_std = value["marker"]["pix"];
      vector<vector<double>> pts3d_std = value["marker"]["pos"];

      // convert pix to normal scale
      for (size_t i = 0; i < pixels_std.size(); ++i) {
        cv::Point2f pix;
        pix.x = width * pixels_std[i][0];
        pix.y = height * pixels_std[i][1];
        pixs.push_back(pix);
      }

      for (size_t i = 0; i < pts3d_std.size(); ++i) {
        cv::Point3d p3d(pts3d_std[i][0], pts3d_std[i][1], pts3d_std[i][2]);
        pts.push_back(p3d);
      }

      names.push_back(name);
      pixels.push_back(pixs);
      pts3d.push_back(pts);
      cameras.push_back(cam);
      sizes.push_back(size);
    }
    return true;
  }
  catch (exception &e) {
    PLOGE << "Exception caught: " << e.what();
    return false;
  }
}

bool ReadCamFromJson(const std::string &filepath, const std::vector<std::string> &names, std::vector<Camera> &cameras)
{
  cameras.clear();
  cameras.resize(names.size());

  auto j = ReadJsonFile(filepath);
  if (j.empty()) {
    PLOGE << "JSON File not exists or cannot be opened: " + filepath;
    return false;
  }

  try {
    auto j_cameras = j["cameras"];
    for (size_t i = 0; i < names.size(); ++i) {
      string rootname, ext;
      splitext(names[i], &rootname, &ext);
      if (j_cameras.contains(rootname)) {
        auto value = j_cameras.at(rootname);
        std::vector<double> vec;
        value["K"].get_to(vec);
        memcpy(cameras[i].K().data, vec.data(), vec.size() * sizeof(double));
        value["R"].get_to(vec);
        memcpy(cameras[i].R().data, vec.data(), vec.size() * sizeof(double));
        value["t"].get_to(vec);
        memcpy(cameras[i].t().data, vec.data(), vec.size() * sizeof(double));
        value["dist"].get_to(vec);
        memcpy(cameras[i].dist().data, vec.data(), vec.size() * sizeof(double));
      }
      else {
        PLOGE << "Cannot find " << rootname << " in file: " << filepath;
        throw runtime_error("Cannot find camera parameters in json file");
      }
    }

    return true;
  }
  catch (exception &e) {
    PLOGE << "Exception caught: " << e.what();
    return false;
  }
}

bool LoadImgsAndFeatures(const std::string &img_dir, const std::string &feature_dir, std::vector<std::string> &fnames,
                         std::vector<ImageFeatures> &features, std::vector<cv::Size> &sizes)
{
  vector<string> fpaths = Listdir(img_dir);
  sort(fpaths.begin(), fpaths.end());

  fnames.clear();
  features.clear();
  sizes.clear();

  for (string &fpath : fpaths) {
    string fname = basename(fpath);
    string rootname, ext;
    splitext(fname, &rootname, &ext);

    static const unordered_set<string> VALID_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"};
    if (VALID_IMG_EXTS.find(ext) == VALID_IMG_EXTS.end())
      continue;

    if (fname == "mask.png")
      continue;

    cv::Mat image = cv::imread(fpath);
    if (image.empty())
      continue;

    ImageFeatures feature;
    feature.img_size = image.size();
    string feature_path = feature_dir + "/" + fname + ".txt";
    ReadColmapFeatures(feature_path, feature.keypoints, feature.descriptors);

    PLOGI << "Index: " << fnames.size() << ", image: " << fname;
    fnames.push_back(fname);
    features.push_back(feature);
    sizes.push_back(image.size());
  }

  int num_images = static_cast<int>(fnames.size());
  if (num_images < 2) {
    PLOGE << "Images number not enough (< 2): " << num_images;
    return false;
  }

  return true;
}

static cv::Mat CalHomography(const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2,
                             const std::vector<cv::DMatch> &matches, double ransac_thresh)
{
  vector<cv::Point2f> ref_pts(matches.size());
  vector<cv::Point2f> src_pts(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    const cv::DMatch &m = matches[i];
    ref_pts[i] = kpts1[m.queryIdx].pt;
    src_pts[i] = kpts2[m.trainIdx].pt;
  }

  vector<char> matches_msk;
  cv::Mat H = cv::findHomography(ref_pts, src_pts, cv::RANSAC, ransac_thresh, matches_msk);

  return H;
}

static float CalMatchingScore(int num_matches, int max_num_matches)
{
  assert(num_matches >= 0 && max_num_matches >= 0);

  if (num_matches >= max_num_matches)
    return 1.0f;
  else
    return static_cast<float>(num_matches) / static_cast<float>(max_num_matches);
}

bool LoadMatchesInfo(const std::string &matches_path, const std::vector<std::string> &fnames,
                     const std::vector<ptzcalib::ImageFeatures> &features, std::vector<ptzcalib::MatchesInfo> &matches_info)
{
  assert(fnames.size() == features.size());

  vector<vector<cv::DMatch>> pairs_matches;
  vector<pair<string, string>> img_pairs_name;
  ReadColmapMatches(matches_path, pairs_matches, img_pairs_name);

  matches_info.clear();
  size_t num_images = fnames.size();
  matches_info.resize(num_images * num_images);

  for (size_t i = 0; i < pairs_matches.size(); ++i) {
    long index_i = FindImgIndex(fnames, img_pairs_name[i].first);
    long index_j = FindImgIndex(fnames, img_pairs_name[i].second);

    static const double RANSAC_THRESH = 4.0;
    cv::Mat H_j_i = CalHomography(features[index_i].keypoints, features[index_j].keypoints, pairs_matches[i], RANSAC_THRESH);

    MatchesInfo match_info_i_j;
    match_info_i_j.matches = pairs_matches[i];
    match_info_i_j.H = H_j_i;
    match_info_i_j.inliers_mask = vector<uchar>(pairs_matches[i].size(), 1);
    match_info_i_j.num_inliers = pairs_matches[i].size();

    static const int MAX_NUM_MATCHES = 100;
    match_info_i_j.confidence = CalMatchingScore(pairs_matches[i].size(), MAX_NUM_MATCHES);

    match_info_i_j.src_img_idx = index_i;
    match_info_i_j.dst_img_idx = index_j;
    matches_info[index_i * num_images + index_j] = match_info_i_j;
  }

  return true;
}

bool LoadAnnotation(const std::string &annot_path, const std::vector<std::string> &fnames, std::vector<std::vector<cv::Point2f>> &pixels,
                    std::vector<std::vector<cv::Point3d>> &pts3d)
{
  vector<string> gt_names;
  vector<vector<cv::Point2f>> gt_pixels;
  vector<vector<cv::Point3d>> gt_pts3d;
  vector<Camera> gt_cameras;
  vector<cv::Size> gt_sizes;

  pixels.clear();
  pts3d.clear();

  bool load_status = ReadFromJson(annot_path, gt_cameras, gt_names, gt_pixels, gt_pts3d, gt_sizes);
  if (!load_status)
    return false;

  size_t num_images = fnames.size();
  pixels.resize(num_images);
  pts3d.resize(num_images);

  for (size_t i = 0; i < gt_cameras.size(); ++i) {
    long idx = FindImgIndex(fnames, gt_names[i]);
    if (idx == -1)
      continue;

    pixels[idx] = gt_pixels[i];
    pts3d[idx] = gt_pts3d[i];
  }

  return true;
}

void SaveRegisteredCam(const std::vector<ptzcalib::Camera> &cameras, const std::unordered_set<long> &reg_image_ids,
                       const std::vector<std::string> &fnames, const std::vector<std::vector<cv::Point2f>> &pixels,
                       const std::vector<std::vector<cv::Point3d>> &pts3d, const std::string &out_path)
{
  vector<Camera> cameras_registered;
  vector<string> names_registered;
  vector<vector<cv::Point2f>> pixels_registered;
  vector<vector<cv::Point3d>> pts3d_registered;

  for (size_t i = 0; i < cameras.size(); ++i) {
    if (reg_image_ids.find(i) == reg_image_ids.end()) {
      PLOGI << "Filter out failed image #" << i << ": " << fnames[i];
      continue;
    }

    cameras_registered.push_back(cameras[i]);
    names_registered.push_back(fnames[i]);
    pixels_registered.push_back(pixels[i]);
    pts3d_registered.push_back(pts3d[i]);
  }

  SaveToJson(cameras_registered, names_registered, pixels_registered, pts3d_registered, out_path);
}

long FindImgIndex(const std::vector<std::string> &fnames, const std::string &fname)
{
  for (size_t i = 0; i < fnames.size(); ++i) {
    string name_wo_ext_i, ext_i;
    splitext(fnames[i], &name_wo_ext_i, &ext_i);

    string name_wo_ext, ext;
    splitext(fname, &name_wo_ext, &ext);

    if (name_wo_ext_i == name_wo_ext)
      return static_cast<long>(i);
  }

  return -1;
}

}  // namespace ptzcalib
