// Used and imported geometric and data structures

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

/* 3D geometry primitive */
struct Triangle {
  vec3 a{};
  vec3 b{};
  vec3 c{};
  vec3 centroid{};
};

/* Bounding volume hierarchy node */
struct BVH {
  BVH* left = nullptr;
  BVH* right = nullptr;
  Triangle* triangle = nullptr;
};

/* 3D vector with origin, direction and length */
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