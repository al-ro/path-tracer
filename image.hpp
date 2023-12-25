#pragma once

#include <cuda_runtime_api.h>

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

class GPUImage {
 public:
  const uint width{0};
  const uint height{0};
  // Device side pointer
  vec3* data;

  GPUImage() = default;
  GPUImage(const Image& image);
  GPUImage(uint width, uint height);
  GPUImage(uint width, uint height, vec3* data);
  ~GPUImage();

  __device__ inline vec3 operator[](uint i) const {
    return data[i];
  }

  __device__ inline vec3& operator[](uint i) {
    return data[i];
  }

  __device__ inline vec3 operator()(vec2 uv) const {
    uv = fract(uv);
    uvec2 coord = vec2(uv) * vec2{width, height};
    uint idx = coord.y * width + coord.x;
    if (idx >= width * height) {
      return vec3(0);
    }
    return data[idx];
  }
};