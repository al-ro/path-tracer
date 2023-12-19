#pragma once

#include <vector>

#include "dataTypes.hpp"

class Geometry {
 public:
  std::vector<Triangle> primitives{};
  std::vector<uint> indices{};
  std::vector<vec3> normals{};
  std::vector<BVHNode> bvh{};
  vec3 center{};

  Geometry() = delete;
  Geometry(const Geometry&) = delete;

  Geometry(std::vector<Triangle> primitives);

  Geometry(Geometry&&) = default;
  ~Geometry() = default;

  void generateIndices();
  void generateNormals();
  void generateBVH();
  // Find the distance to the closest intersection, the index of the primitive and the number of BVH tests.
  // FLT_MAX if no intersection
  void intersect(Ray& ray, uint& index, uint& count);
};

vec3 getNormal(const Triangle&);