#pragma once

#include <vector>

#include "dataTypes.hpp"

class Geometry {
 public:
  std::vector<Triangle> primitives{};
  std::vector<uint> indices{};
  std::vector<vec3> normals{};
  VertexAttributes attributes{};
  std::vector<BVHNode> bvh{};
  vec3 aabbMin{FLT_MAX};
  vec3 aabbMax{-FLT_MAX};
  std::vector<vec3> corners{8};
  vec3 centroid{0};

  Geometry() = delete;
  Geometry(const Geometry&) = delete;

  Geometry(std::vector<Triangle> primitives, VertexAttributes attributes);

  Geometry(Geometry&&) = default;
  ~Geometry() = default;

  vec2 getTexCoord(uint idx, vec2 barycentric) const;
  vec3 getNormal(uint idx, vec2 barycentric) const;

  void generateIndices();
  void generateNormals();
  void generateBVH();
  // Find the distance to the closest intersection, the index of the primitive and the number of BVH tests.
  void intersect(Ray& ray, HitRecord& hitRecord, uint& count) const;
};
