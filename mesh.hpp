#pragma once

#include <memory>

#include "dataTypes.hpp"
#include "geometry.hpp"
#include "material.hpp"

/*
    A class representing an instance of a Geometry with an applied Material and spatial transformation
*/
class Mesh {
 public:
  // Reference to an externally managed complete Geometry.
  std::shared_ptr<Geometry> geometry;

  // Reference to an externally managed Material.
  std::shared_ptr<Material> material;

  vec3 translation{0};
  vec3 rotation{0};
  float scale{1};

  // Translation, rotation, scale
  mat4 modelMatrix = identity<mat4>();

  // Inverse of the model matrix to transform rays
  mat4 invModelMatrix = identity<mat4>();

  mat4 normalMatrix = transpose(invModelMatrix);

  // Transformed limits for constructing TLAS
  vec3 aabbMin{FLT_MAX};
  vec3 aabbMax{-FLT_MAX};
  vec3 centroid{0};

  Mesh() = default;
  Mesh(std::shared_ptr<Geometry> geometry, std::shared_ptr<Material> material);
  ~Mesh() = default;

  void setPosition(vec3 translation);
  void setRotationX(float angle);
  void setRotationY(float angle);
  void setRotationZ(float angle);
  void setScale(float scale);
  void update();

  vec3 getMin() const;
  vec3 getMax() const;

  // Find the distance to the closest intersection, the index of the primitive and the number of BVH tests.
  // Store distance in ray.t  (FLT_MAX if no intersection) and geometry data in hitRecord
  void intersect(Ray& ray, HitRecord& hitRecord, uint& count) const;

  // Center geometry at the origin
  void centerGeometry();
};

class GPUMesh {
 public:
  // Inverse of the model matrix to transform rays
  mat4 invModelMatrix = identity<mat4>();
  // Device pointer to an externally managed complete GPUGeometry.
  GPUGeometry* geometry = nullptr;

  // Device pointer to an externally managed GPUMaterial.
  GPUMaterial* material = nullptr;

  // mat4 normalMatrix = transpose(invModelMatrix);

  GPUMesh() = delete;
  GPUMesh(const Mesh& mesh, GPUGeometry* geometry, GPUMaterial* material);
  ~GPUMesh() = default;

  // Find the distance to the closest intersection, the index of the primitive and the number of BVH tests.
  // Store distance in ray.t  (FLT_MAX if no intersection) and geometry data in hitRecord
  __device__ void intersect(Ray& ray, HitRecord& hitRecord, uint& count) const;
};
