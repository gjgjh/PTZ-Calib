/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-18 10:17:28
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-02-18 10:18:03
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#ifndef PTZ_CALIB_SRC_CORE_FLAT_PAIR_MAP_H
#define PTZ_CALIB_SRC_CORE_FLAT_PAIR_MAP_H

#include <algorithm>
#include <utility>
#include <vector>

namespace ptzcalib {

/// Lightweight copy of the flat_map of BOOST library
/// Use a vector to speed up insertion (preallocated array)
template <typename T1, typename T2>
class flat_pair_map {
  using P = std::pair<T1, T2>;

 public:
  using iterator = typename std::vector<P>::iterator;

  typename std::vector<P>::iterator find(const T1 &val) { return std::lower_bound(m_vec.begin(), m_vec.end(), val, superiorToFirst); }

  T2 &operator[](const T1 &val) { return std::lower_bound(m_vec.begin(), m_vec.end(), val, superiorToFirst)->second; }

  void sort() { std::sort(m_vec.begin(), m_vec.end(), sortPairAscend); }
  void push_back(const P &val) { m_vec.push_back(val); }
  void clear() { m_vec.clear(); }
  void reserve(size_t count) { m_vec.reserve(count); }

  template <class... Args>
  void emplace_back(Args &&...args)
  {
    m_vec.emplace_back(std::forward<Args>(args)...);
  }

  size_t size() const { return m_vec.size(); }
  const P &operator[](std::size_t idx) const { return m_vec[idx]; }

 private:
  std::vector<P> m_vec;

  static bool sortPairAscend(const P &a, const P &b) { return a.first < b.first; }
  static bool superiorToFirst(const P &a, const T1 &b) { return a.first < b; }
};

}  // namespace ptzcalib

#endif  // PTZ_CALIB_SRC_CORE_FLAT_PAIR_MAP_H
