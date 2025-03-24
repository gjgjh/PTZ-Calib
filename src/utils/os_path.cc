/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 16:45:58
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-02-17 19:57:19
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#include "os_path.h"

#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

#include "filesystem.h"

using namespace std;
namespace fs = ghc::filesystem;

bool exists(const string& path)
{
#if _WIN32
  DWORD ftyp = GetFileAttributesA(path.c_str());
  if (ftyp == INVALID_FILE_ATTRIBUTES)
    return false;  // something is wrong with your path!

  if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
    return true;  // this is a directory!
  return false;   // this is not a directory!
#else
  return access(path.c_str(), F_OK) == 0;
#endif
}

bool mkdir_ifnot_exist(const std::string& dir)
{
  if (exists(dir))
    return true;
#if _WIN32
  if ((GetFileAttributes(dir.c_str())) == INVALID_FILE_ATTRIBUTES) {
    return CreateDirectory(dir.c_str(), 0);
  }
#else
  if (-1 == mkdir(dir.c_str(), 0755)) {
    cerr << "mkdir " << dir << " failed";
    return false;
  }
#endif
  return true;
}

bool mkdirs(const string& dir)
{
  if (dir.empty())
    return true;

  const char* p = dir.c_str();
  const char* q = p + 1;
  string path;
  while (*q != '\0') {
    if (*q == '/' || *q == '\\') {
      path.append(p, q);
      if (false == mkdir_ifnot_exist(path)) {
        cerr << "mkdir " << path << " failed";
        return false;
      }
      p = q;
      while (*q == '/' || *q == '\\') {
        ++q;
      }
    }
    else {
      ++q;
    }
  }
  path.append(p, q);
  if (false == mkdir_ifnot_exist(path)) {
    cerr << "mkdir " << path << " failed";
    return false;
  }

  return true;
}

void list_files(const std::string& path, vector<string>& paths)
{
  struct stat s;
  if (stat(path.c_str(), &s) == 0) {
    if (s.st_mode & S_IFREG) {
      paths.push_back(path);
    }
    else if (s.st_mode & S_IFDIR) {
      DIR* dir;
      struct dirent* ent;
      if ((dir = opendir(path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
          std::string found(ent->d_name, strlen(ent->d_name));
          if (found != "." && found != "..") {
            list_files(join_path(path, found), paths);
          }
        }
        closedir(dir);
      }
    }
  }
}

std::vector<std::string> list_files2(const std::string& path)
{
  std::vector<std::string> paths;
  struct stat s;
  if (stat(path.c_str(), &s) == 0) {
    if (s.st_mode & S_IFREG) {
      paths.push_back(path);
    }
  }
  return paths;
}

std::vector<std::string> Listdir(const std::string& dir)
{
  std::vector<std::string> paths;
  if (!exists(dir))
    return paths;

  for (const auto& entry : fs::directory_iterator(dir)) paths.push_back(entry.path());
  return paths;
}

std::string dirname(const std::string& path)
{
  const char* pbegin = path.c_str();
  const char* pend = pbegin + path.length();
  const char* pdata = pend - 1;
  while (pdata >= pbegin) {
    if (*pdata == '/' || *pdata == '\\') {
      break;
    }
    --pdata;
  }
  const char* sep = pdata;
  if (sep < pbegin) {
    return "";
  }
  else if (sep == pbegin) {
    return path.substr(0, 1);
  }
  else {
    --pdata;
    while (pdata >= pbegin) {
      if (*pdata != '/' && *pdata != '\\') {
        return string(pbegin, pdata + 1);
      }
      --pdata;
    }
    return string(pbegin, sep + 1);
  }
}

std::string basename(const std::string& path)
{
  size_t found = path.find_last_of("/\\");
  if (string::npos == found) {
    return path;
  }
  else {
    return path.substr(found + 1);
  }
}

long last_modified_time(const string& path)
{
  struct stat t_stat;
  auto err = stat(path.c_str(), &t_stat);
  if (err == 0) {
#if defined(__APPLE__)
    return t_stat.st_mtimespec.tv_sec;
#else
    return t_stat.st_mtime;
#endif
  }
  else {
    return -1;
  }
}

void splitext(const std::string& path, std::string* filepath, std::string* ext)
{
  const char* pbegin = path.c_str();
  const char* pend = pbegin + path.length();
  const char* pdata = pend - 1;
  while (pdata >= pbegin) {
    if (*pdata == '/' || *pdata == '\\') {
      *filepath = path;
      *ext = "";
      return;
    }
    else if (*pdata == '.') {
      const char* sep = pdata;
      if (sep == pbegin) {
        *filepath = path;
        *ext = "";
        return;
      }
      else {
        --pdata;
        while (pdata >= pbegin) {
          if (*pdata != '.') {
            break;
          }
          --pdata;
        }
        if (pdata < pbegin) {
          *filepath = path;
          *ext = "";
          return;
        }
        else {
          *filepath = string(pbegin, sep);
          *ext = string(sep, pend);
          return;
        }
      }
    }
    --pdata;
  }
  *filepath = path;
  *ext = "";
}

string join_path(const std::string& root_dir, const std::string& filename)
{
  string path;
  if (endswith(root_dir, "/") || endswith(root_dir, "\\")) {
    path = root_dir + filename;
  }
  else {
    path = root_dir + "/" + filename;
  }
  return path;
}