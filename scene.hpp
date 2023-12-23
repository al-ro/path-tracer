#pragma once

#include <iostream>

#include "dataTypes.hpp"
#include "intersection.hpp"
#include "mesh.hpp"

class Scene {
 public:
  std::vector<Mesh> meshes;
  std::vector<uint> indices;
  std::vector<BVHNode> tlas;

  Scene() = default;
  Scene(const std::vector<Mesh>& meshes);
  Scene(const Scene& scene) = delete;
  ~Scene() = default;

  void completeScene();
  void generateIndices();
  void generateTLAS();

  uint intersect(Ray& ray, HitRecord& hitRecord, uint& count) const;
};