// Used and imported geometric and data structures

#pragma once

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

using namespace glm;

/* 2D section of render target */
struct Extent {
  vec2 min{};
  vec2 max{};
};

/* Axis aligned bounding box */
struct AABB {
  vec3 min{};
  vec3 max{};
};
struct Triangle {
  vec3 v0{};
  vec3 v1{};
  vec3 v2{};
  // Location for spatial sorting
  vec3 centroid{};
};

/* Bounding volume hierarchy node */
struct BVH {
  BVH* left = nullptr;
  BVH* right = nullptr;
  Triangle* triangle = nullptr;
};

/* 3D vector with origin, direction and intersection position t */
struct Ray {
  vec3 origin{};
  vec3 direction{};
  // Distance along ray where an intersection occurs
  float t{};
};

struct Camera {
  vec3 position{};
  // Location camera is looking at
  vec3 target{};
  vec3 up{};
  // FOV in degrees
  float fieldOfView{};
};