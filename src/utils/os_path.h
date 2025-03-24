/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 16:17:48
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-02-17 16:46:06
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#ifndef PTZ_CALIB_SRC_UTILS_OS_PATH_H
#define PTZ_CALIB_SRC_UTILS_OS_PATH_H

#include <string>
#include <vector>

bool exists(const std::string& path);
bool mkdir_ifnot_exist(const std::string& dir);
bool mkdirs(const std::string& dir);

void list_files(const std::string& path, std::vector<std::string>& paths);
std::vector<std::string> list_files2(const std::string& path);
std::vector<std::string> Listdir(const std::string& dir);

std::string dirname(const std::string& path);
std::string basename(const std::string& path);

long last_modified_time(const std::string& path);

void splitext(const std::string& path, std::string* filepath, std::string* ext);

std::string join_path(const std::string& root_dir, const std::string& filename);
template <typename... Args>
std::string join_path(const std::string& part1, const std::string& part2, Args... args)
{
  return join_path(part1, join_path(part2, args...));
}

inline bool endswith(const std::string& str, const std::string& prefix)
{
  return (prefix.size() <= str.size()) && (str.compare(str.length() - prefix.length(), prefix.length(), prefix) == 0);
}

#endif  // PTZ_CALIB_SRC_UTILS_OS_PATH_H