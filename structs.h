// Used and imported geometric and data structures

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

/* 2D section of render target */
struct Extent {
  glm::vec2 min{};
  glm::vec2 max{};
};

/* Axis aligned bounding box */
struct AABB {
  glm::vec3 min{};
  glm::vec3 max{};
};

/* 3D geometry primitive */
struct Triangle {
  glm::vec3 a{};
  glm::vec3 b{};
  glm::vec3 c{};
  glm::vec3 centroid{};
};

/* Bounding volume hierarchy node */
struct BVH {
  BVH* left = nullptr;
  BVH* right = nullptr;
  Triangle* triangle = nullptr;
};

/* 3D vector with origin, direction and length */
struct Ray {
  glm::vec3 origin{};
  glm::vec3 direction{};
  float t{};
};