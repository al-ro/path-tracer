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

class GPUScene {
 public:
  GPUMesh* meshes;
  uint* indices;
  BVHNode* tlas;

  GPUScene() = delete;
  GPUScene(const Scene& scene, const std::vector<GPUMesh>& gpuMeshes);
  // Disallow copies for objects managing GPU memory
  GPUScene(const GPUScene& scene) = delete;
  ~GPUScene();
  __device__ uint intersect(Ray& ray, HitRecord& hitRecord, uint& count) const {
    return intersectTLAS(ray, tlas, meshes, indices, hitRecord, count);
  }
};