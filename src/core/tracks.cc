/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-18 10:16:08
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-02-18 10:24:34
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#include "tracks.h"

#include <fstream>
#include <iostream>

using namespace std;

namespace ptzcalib {

void TracksBuilder::Build(const vector<MatchesInfo>& matches_info)
{
  // 1. Calculate how much single set we will have
  // each set is made of a tuple : (imageIndex, featureIndex)
  set<IndexedFeaturePair> all_features;
  for (auto& match_info : matches_info) {
    int i = match_info.src_img_idx;
    int j = match_info.dst_img_idx;

    for (auto& match : match_info.matches) {
      all_features.emplace(i, match.queryIdx);
      all_features.emplace(j, match.trainIdx);
    }
  }

  // 2. Build the 'flat' representation where a tuple (the node)
  //  is attached to a unique index.
  map_node_to_index_.reserve(all_features.size());
  int count = 0;
  for (auto& feat : all_features) {
    map_node_to_index_.emplace_back(feat, count);
    ++count;
  }
  map_node_to_index_.sort();
  all_features.clear();

  // 3. Add the node and the pairwise correspondences in the UF tree.
  uf_tree_.InitSets(static_cast<int>(map_node_to_index_.size()));

  // 4. Union of the matched features corresponding UF tree sets
  for (auto& match_info : matches_info) {
    int i = match_info.src_img_idx;
    int j = match_info.dst_img_idx;

    for (auto& match : match_info.matches) {
      IndexedFeaturePair pair_i(i, match.queryIdx);
      IndexedFeaturePair pair_j(j, match.trainIdx);
      int index_i = map_node_to_index_[pair_i];
      int index_j = map_node_to_index_[pair_j];
      uf_tree_.Union(index_i, index_j);
    }
  }
}

void TracksBuilder::Filter(int min_track_length)
{
  // Build the Track observations & mark tracks that have id collision:
  map<int, set<int>> tracks;      // {track_id, {image_id, image_id, ...}}
  set<int> problematic_track_id;  // {track_id, ...}

  // For each node retrieve its track id from the UF tree and add the node to the track
  // - if an image id is observed multiple time, then mark the track as invalid
  //   - a track cannot list many times the same image index
  for (int k = 0; k < map_node_to_index_.size(); ++k) {
    const int& track_id = uf_tree_.Find(k);
    const auto& feat = map_node_to_index_[k];

    // Augment the track and mark if invalid (an image can only be listed once)
    if (tracks[track_id].insert(feat.first.first).second == false) {
      problematic_track_id.insert(track_id);  // invalid
    }
  }

  // Reject tracks that have too few observations
  for (const auto& val : tracks) {
    if (val.second.size() < min_track_length) {
      problematic_track_id.insert(val.first);
    }
  }

  // Reset the marked invalid track ids in the UF Tree
  for (int& root_index : uf_tree_.m_cc_parent) {
    if (problematic_track_id.count(root_index) > 0) {
      // reset selected root
      uf_tree_.m_cc_size[root_index] = 1;
      root_index = numeric_limits<int>::max();
    }
  }
}

void TracksBuilder::ExportToSTL(Tracks& tracks)
{
  tracks.clear();
  for (int k = 0; k < map_node_to_index_.size(); ++k) {
    const auto& feat = map_node_to_index_[k];
    const int& track_id = uf_tree_.m_cc_parent[k];

    if (  // ensure never add rejected elements (track marked as invalid)
        track_id != numeric_limits<int>::max()
        // ensure never add 1-length track element (it's not a track)
        && uf_tree_.m_cc_size[track_id] > 1) {
      tracks[track_id].insert(feat.first);
    }
  }
}

size_t TracksBuilder::NbTracks() const
{
  std::set<int> parent_id(uf_tree_.m_cc_parent.cbegin(), uf_tree_.m_cc_parent.cend());
  // Erase the "special marker" that depicted rejected tracks
  parent_id.erase(std::numeric_limits<int>::max());
  return parent_id.size();
}

void Length(const Tracks& tracks, int& total_length, int& max_length, int& min_length)
{
  total_length = 0;
  max_length = 0;
  min_length = INT_MAX;

  for (auto& track_elem : tracks) {
    int track_id = track_elem.first;
    const Track& track = track_elem.second;

    total_length += static_cast<int>(track.size());
    max_length = max(max_length, static_cast<int>(track.size()));
    min_length = min(min_length, static_cast<int>(track.size()));
  }
}

static int isConnected(int img_id, const vector<set<int>>& connected_img_sets)
{
  for (size_t i = 0; i < connected_img_sets.size(); ++i) {
    if (connected_img_sets[i].count(img_id) > 0) {
      return static_cast<int>(i);
    }
  }

  return -1;
}

void FindMaxCoVisible(const Tracks& tracks, int num_images, set<int>& max_connect_imgs)
{
  // Each is a independent connected (co-visible) component
  vector<set<int>> connected_img_sets;

  for (auto& track_elem : tracks) {
    int track_id = track_elem.first;
    const Track& track = track_elem.second;

    bool is_not_connect = true;
    set<int, greater<int>> connected_idx_all;  // Sort from large to small
    for (auto& kv : track) {
      int img_id = kv.first;

      int connected_idx = isConnected(img_id, connected_img_sets);
      if (connected_idx != -1) {
        is_not_connect = false;
        connected_idx_all.insert(connected_idx);
      }
    }

    set<int> co_visible_new;
    for (auto& kv : track) {
      int img_id = kv.first;
      co_visible_new.insert(img_id);
    }

    if (is_not_connect) {
      // Create a new co-visible set
      connected_img_sets.push_back(co_visible_new);
    }
    else {
      // Merge all connected into one co-visible set
      for (auto& idx : connected_idx_all) {
        co_visible_new.insert(connected_img_sets[idx].begin(), connected_img_sets[idx].end());
      }
      connected_img_sets.push_back(co_visible_new);

      for (auto& idx : connected_idx_all) {
        connected_img_sets.erase(connected_img_sets.begin() + idx);
      }
    }
  }

  int max_connect_imgs_num = 0;
  max_connect_imgs.clear();
  for_each(connected_img_sets.begin(), connected_img_sets.end(), [&](const set<int>& co_visible) {
    if (co_visible.size() > max_connect_imgs_num) {
      max_connect_imgs_num = static_cast<int>(co_visible.size());
      max_connect_imgs = co_visible;
    }
  });
}

void SaveTracks(const Tracks& tracks, const std::vector<std::string>& img_names, const std::string& outpath)
{
  try {
    ofstream fout(outpath);

    if (fout.is_open()) {
      for (const auto& kv : tracks) {
        int track_id = kv.first;
        const Track& track = kv.second;

        string delim = "";
        for (const auto& kv2 : track) {
          int img_id = kv2.first;
          string img_name = img_names[img_id];
          int feature_id = kv2.second;

          fout << delim << img_name << " " << feature_id;
          delim = " ";
        }

        fout << endl;
      }
    }
    else {
      throw runtime_error("Write file error");
    }

    fout.close();
  }
  catch (exception& e) {
    cerr << "Cannot write file to " << outpath << endl;
  }
}

}  // namespace ptzcalib
