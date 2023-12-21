#pragma once

#include <vector>

#include "dataTypes.hpp"

class Image {
 public:
  uint width{0};
  uint height{0};
  std::vector<vec3> data{};

  Image() = default;
  Image(uint width, uint height);
  Image(uint width, uint height, const std::vector<vec3>& data);

  inline vec3 operator[](uint i) const {
    return data[i];
  }

  inline vec3& operator[](uint i) {
    return data[i];
  }

  inline vec3 operator()(vec2 uv) const {
    uv = fract(uv);
    uvec2 coord = vec2(uv) * vec2{width, height};
    uint idx = coord.y * width + coord.x;
    if (idx >= data.size()) {
      return vec3(0);
    }
    return data[idx];
  }
};