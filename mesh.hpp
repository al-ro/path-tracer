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
  /*
    Reference to an externally managed complete Geometry.
    A Mesh cannot be created without a Geometry and cannot have its geometry changed.
  */
  const Geometry& geometry;
  Material material;
  // Translation, rotation, scale
  mat4 modelMatrix = identity<mat4>();
  // Inverse of the model matrix to transform rays
  mat4 invModelMatrix = identity<mat4>();
  mat4 normalMatrix = transpose(invModelMatrix);
  AABB transformedAABB{};

  Mesh() = delete;
  Mesh(const Geometry&, Material);
  Mesh(const Mesh&) = delete;
  Mesh(Mesh&&) = default;
  ~Mesh() = default;

  void translate(vec3 translation);
  void rotateX(float angle);
  void rotateY(float angle);
  void rotateZ(float angle);
  void scale(float scale);
  void update();

  // Center geometry at the origin
  void center();
};
