// Used and imported geometric and data structures

#pragma once

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_ENABLE_EXPERIMENTAL

#include <cuda_runtime_api.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <iostream>
#include <vector>

using namespace glm;

// if NVCC
#if defined(__CUDACC__)
#define ALIGN(n) __align__(n)
#else
#define ALIGN(n) __attribute__((aligned(n)))
#endif

// Axis aligned bounding box
struct AABB {
  vec3 min{FLT_MAX};
  vec3 max{-FLT_MAX};

  // Increase AABB dimensions by including new point p
  inline void grow(vec3 p) {
    min = glm::min(min, p);
    max = glm::max(max, p);
  }

  // Increase AABB dimensions by including new AABB b
  inline void grow(AABB b) {
    min = glm::min(min, b.min);
    max = glm::max(max, b.max);
  }

  // Return surface area 2xy * 2xz * 2yz
  inline float area() const {
    vec3 dim = max - min;
    return 2.0f * (dim.x * dim.y + dim.y * dim.z + dim.z * dim.x);
  }
};

struct Triangle {
  vec3 v0{};
  vec3 v1{};
  vec3 v2{};
  // Location for spatial sorting
  vec3 centroid{FLT_MAX};
  inline vec3 getMin() const { return min(min(v0, v1), v2); }
  inline vec3 getMax() const { return max(max(v0, v1), v2); }
};

struct GPUTriangle {
  vec3 v0{};
  vec3 v1{};
  vec3 v2{};
};

// Per-vertex normals and texture coordinates
struct VertexAttributes {
  std::vector<vec3> normals{};
  std::vector<vec2> texCoords{};
};

/* Bounding volume hierarchy node */
struct BVHNode {
  AABB aabb;
  // Index of first primitive or left child
  uint leftFirst;
  // Number of primitives
  uint count;
};

// The extent of the primitives in a BVH construction interval and the number of primitives in it
struct Bin {
  AABB bounds{};
  uint count{};
};

// 3D vector with origin, direction and intersection position t
struct Ray {
  vec3 origin{0};
  vec3 direction{0, 0, -1};
  vec3 invDirection{0, 0, -1};
  // Distance along ray where an intersection occurs
  float t{FLT_MAX};
};

// Stores primitive index and barycentric coordinates of intersection point
struct HitRecord {
  vec2 barycentric{0};
  uint hitIndex{UINT_MAX};
};

struct Camera {
  vec3 position{};
  // Location camera is looking at
  vec3 target{};
  vec3 up{};
  // FOV in degrees
  float fieldOfView{};
};

inline std::ostream& operator<<(std::ostream& os, const vec3& v) {
  for (uint i = 0; i < 3; i++) {
    os << v[i] << " ";
  }

  return os;
}

inline std::ostream& operator<<(std::ostream& os, const mat4x4& m) {
  for (uint i = 0; i < 4; i++) {
    for (uint j = 0; j < 4; j++) {
      os << m[i][j] << "\t";
    }
    os << "\n";
  }

  return os;
}

template <typename T>
void inline reorder(std::vector<T>& values, const std::vector<uint>& order, const uint size = 1u) {
  std::vector<T> reorderedValues(values.size());
  for (int i = 0; i < order.size(); i++) {
    for (int j = 0; j < size; j++) {
      reorderedValues[size * i + j] = values[size * order[i] + j];
    }
  }
  values = reorderedValues;
}
__host__ __device__ inline float dot_c(const vec3& a, const vec3& b) {
  return max(dot(a, b), 1e-5f);
}