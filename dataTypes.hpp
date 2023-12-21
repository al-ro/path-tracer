// Used and imported geometric and data structures

#pragma once

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <iostream>
#include <vector>

using namespace glm;

/* 2D section of render target */
struct Extent {
  vec2 min{};
  vec2 max{};
};

/* Axis aligned bounding box */
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

/* Bounding volume hierarchy node */
struct BVHNode {
  vec3 aabbMin;
  vec3 aabbMax;
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

/* 3D vector with origin, direction and intersection position t */
struct Ray {
  vec3 origin{};
  vec3 direction{};
  vec3 invDirection{};
  // Distance along ray where an intersection occurs
  float t{FLT_MAX};
};

struct HitRecord {
  uint hitIndex{UINT_MAX};
  float dist{FLT_MAX};
  vec3 normal{0};
};

struct Camera {
  vec3 position{};
  // Location camera is looking at
  vec3 target{};
  vec3 up{};
  // FOV in degrees
  float fieldOfView{};
};

struct Image {
  uint width{};
  uint height{};
  std::vector<vec3> data{width * height};

  inline vec3 operator[](uint i) const {
    return data[i];
  }
  inline vec3& operator[](uint i) {
    return data[i];
  }
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