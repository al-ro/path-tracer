#pragma once

#include <cuda_runtime_api.h>

#include <vector>

#include "dataTypes.hpp"

class Geometry {
 public:
  std::vector<Triangle> primitives{};
  std::vector<uint> indices{};
  std::vector<vec3> faceNormals{};
  VertexAttributes attributes{};
  std::vector<BVHNode> bvh{};
  vec3 aabbMin{FLT_MAX};
  vec3 aabbMax{-FLT_MAX};
  std::vector<vec3> corners{8};
  vec3 centroid{0};

  Geometry() = delete;
  Geometry(std::vector<Triangle> primitives, VertexAttributes attributes = VertexAttributes{});
  Geometry(const Geometry& geometry) = delete;
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

class GPUGeometry {
 public:
  GPUTriangle* primitives;
  vec3* faceNormals;
  vec3* vertexNormals;
  vec2* texCoords;
  BVHNode* bvh;

  bool hasNormals{false};
  bool hasTexCoords{false};

  GPUGeometry() = delete;
  GPUGeometry(const Geometry& geometry);
  // Delete copy constructor as the object manages its data on the GPU
  GPUGeometry(const GPUGeometry& geometry) = delete;
  GPUGeometry(GPUGeometry&& geometry) = default;

  ~GPUGeometry();

  __device__ vec2 getTexCoord(uint idx, vec2 barycentric) const;

  __device__ vec3 getNormal(uint idx, vec2 barycentric) const;

  // Find the distance to the closest intersection, the index of the primitive and the number of BVH tests.
  __device__ void intersect(Ray& ray, HitRecord& hitRecord, uint& count) const;
};
