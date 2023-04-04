//    Copyright 2023 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace openjij {
namespace utility {
struct UnionFind {
  using Node = std::size_t;
  using Parent = std::vector<Node>;
  using Rank = std::vector<Node>;
  using size_type = Parent::size_type;

  explicit UnionFind(size_type n) : _parent(n), _rank(n, 0) {
    std::iota(_parent.begin(), _parent.end(), 0);
  }

  void unite_sets(Node x, Node y) {
    auto root_x = find_set(x);
    auto root_y = find_set(y);

    if (root_x == root_y)
      return;

    if (_rank[root_x] > _rank[root_y]) {
      _parent[root_y] = root_x;
    } else {
      _parent[root_x] = root_y;
      if (_rank[root_x] == _rank[root_y])
        ++_rank[root_y];
    }
  }

  Node find_set(Node node) {
    auto base_node = node;

    // find root node
    auto parent_node = _parent[node];
    while (parent_node != node) {
      node = parent_node;
      parent_node = _parent[node];
    }
    auto root_node = parent_node;

    // do path compression
    node = _parent[base_node];
    while (root_node != node) {
      _parent[base_node] = root_node;
      base_node = node;
      node = _parent[base_node];
    }
    return root_node;
  }

  Parent get_roots() const {
    auto roots_list = _parent;
    std::sort(roots_list.begin(), roots_list.end());
    roots_list.erase(std::unique(roots_list.begin(), roots_list.end()),
                     roots_list.end());
    return roots_list;
  }

private:
  Parent _parent;
  Rank _rank;
};
} // namespace utility
} // namespace openjij
