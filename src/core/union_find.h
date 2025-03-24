/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-18 10:23:03
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-02-18 10:23:56
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#ifndef PTZ_CALIB_SRC_CORE_UNION_FIND_H
#define PTZ_CALIB_SRC_CORE_UNION_FIND_H

#include <numeric>
#include <vector>

namespace ptzcalib {

// Union-Find/Disjoint-Set data structure.
// ref: https://github.com/openMVG/openMVG/blob/master/src/openMVG/tracks/union_find.hpp
//--
// A disjoint-set data structure also called a union–find data structure
// or merge–find set, is a data structure that keeps track of a set of elements
// partitioned into a number of disjoint (non-overlapping) subsets.
// It supports two operations:
// - Find: Determine which subset a particular element is in.
//   - It returns an item from this set that serves as its "representative";
// - Union: Join two subsets into a single subset.
// Sometime a Connected method is implemented:
// - Connected:
//   - By comparing the result of two Find operations, one can determine whether
//      two elements are in the same subset.
//--
struct UnionFind {
  // Represent the DS/UF forest thanks to two array:
  // A parent 'pointer tree' where each node holds a reference to its parent node
  std::vector<int> m_cc_parent;
  // A rank array used for union by rank
  std::vector<int> m_cc_rank;
  // A 'size array' to know the size of each connected component
  std::vector<int> m_cc_size;

  // Init the UF structure with num_cc nodes
  void InitSets(const int num_cc)
  {
    // all set size are 1 (independent nodes)
    m_cc_size.resize(num_cc, 1);
    // Parents id have their own CC id {0,n}
    m_cc_parent.resize(num_cc);
    std::iota(m_cc_parent.begin(), m_cc_parent.end(), 0);
    // Rank array (0)
    m_cc_rank.resize(num_cc, 0);
  }

  // Return the number of nodes that have been initialized in the UF tree
  int GetNumNodes() const { return static_cast<int>(m_cc_size.size()); }

  // Return the representative set id of I nth component
  int Find(int i)
  {
    if (i < 0 || i >= static_cast<int>(m_cc_parent.size())) {
      throw std::out_of_range("Index out of range");
    }

    // Recursively set all branch as children of root (Path compression)
    if (m_cc_parent[i] != i)
      m_cc_parent[i] = Find(m_cc_parent[i]);
    return m_cc_parent[i];
  }

  // Replace sets containing I and J with their union
  void Union(int i, int j)
  {
    int root_i = Find(i);
    int root_j = Find(j);

    if (root_i == root_j) {
      // Already in the same set. Nothing to do
      return;
    }

    // x and y are not already in same set. Merge them.
    // Perform an union by rank:
    //  - always attach the smaller tree to the root of the larger tree
    if (m_cc_rank[root_i] < m_cc_rank[root_j]) {
      m_cc_parent[root_i] = root_j;
      m_cc_size[root_j] += m_cc_size[root_i];
    }
    else {
      m_cc_parent[root_j] = root_i;
      m_cc_size[root_i] += m_cc_size[root_j];
      if (m_cc_rank[root_i] == m_cc_rank[root_j]) {
        ++m_cc_rank[root_i];
      }
    }
  }

  // Check if elements i and j are in the same set
  bool Connected(int i, int j) { return Find(i) == Find(j); }

  // Get the size of the connected component containing i
  int Size(int i)
  {
    int root = Find(i);
    return m_cc_size[root];
  }
};

}  // namespace ptzcalib

#endif  // PTZ_CALIB_SRC_CORE_UNION_FIND_H