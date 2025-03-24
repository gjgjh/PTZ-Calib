/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-18 10:16:05
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-02-18 10:24:27
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#ifndef PTZ_CALIB_SRC_CORE_TRACKS_H
#define PTZ_CALIB_SRC_CORE_TRACKS_H

#include <map>
#include <set>
#include <vector>

#include "flat_pair_map.h"
#include "types.h"
#include "union_find.h"

namespace ptzcalib {

/** {ImageId, FeatureId} */
using IndexedFeaturePair = std::pair<int, int>;

/** Data structure to store a track: collection of {ImageId, FeatureId}
 * The corresponding image points with their imageId and FeatureId.
 */
using Track = std::map<int, int>;

/** Tracks are the collection of {TrackId, Track} */
using Tracks = std::map<int, Track>;

/** Tracks builder
 * ref: https://github.com/openMVG/openMVG/blob/master/src/openMVG/tracks/tracks.hpp */
class TracksBuilder {
 public:
  TracksBuilder() = default;
  ~TracksBuilder() = default;

  void Build(const std::vector<MatchesInfo>& matches_info);

  /** Remove bad tracks (too short or track with ids collision) */
  void Filter(int min_track_length = 2);

  /** Export tracks as a map (each entry is a sequence of ImageId and FeatureId):
   * {TrackId => {(ImageId, FeatureId), ... ,(ImageId, FeatureId)}
   */
  void ExportToSTL(Tracks& tracks);

  // Return the number of connected set in the UnionFind structure (tree forest)
  size_t NbTracks() const;

  flat_pair_map<IndexedFeaturePair, int> map_node_to_index_;
  UnionFind uf_tree_;
};

/* Calculate tracks length */
void Length(const Tracks& tracks, int& total_length, int& max_length, int& min_length);

/** Find max co-visible images set based on tracks */
void FindMaxCoVisible(const Tracks& tracks, int num_images, std::set<int>& max_connect_imgs);

/* Save tracks to file */
void SaveTracks(const Tracks& tracks, const std::vector<std::string>& img_names, const std::string& outpath);

}  // namespace ptzcalib

#endif  // PTZ_CALIB_SRC_CORE_TRACKS_H