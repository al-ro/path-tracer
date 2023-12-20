#include "mesh.hpp"

#include <iostream>

Mesh::Mesh(const Geometry& geometry, Material material) : geometry{geometry}, material{material} {
  update();
}

HitRecord Mesh::intersect(const Ray& ray, uint& count) const {
  Ray transformedRay = ray;
  transformedRay.origin = invModelMatrix * vec4(ray.origin, 1.0f);
  // Not normalized to handle scale transform
  transformedRay.direction = invModelMatrix * vec4(ray.direction, 0.0f);
  transformedRay.invDirection = 1.0f / transformedRay.direction;

  uint hitIndex{UINT_MAX};

  // Hit distance is recorded in .t of the ray passed in
  geometry.intersect(transformedRay, hitIndex, count);

  vec3 normal{0};

  if (hitIndex < UINT_MAX) {
    normal = geometry.normals[hitIndex];
  }

  return HitRecord{hitIndex, transformedRay.t, normal};
}

void Mesh::update() {
  invModelMatrix = inverse(modelMatrix);
  normalMatrix = transpose(invModelMatrix);

  transformedAABBMin = vec3{FLT_MAX};
  transformedAABBMax = vec3{-FLT_MAX};

  for (auto& corner : geometry.corners) {
    transformedAABBMin = min(transformedAABBMin, vec3(modelMatrix * vec4(corner, 1.0f)));
    transformedAABBMax = max(transformedAABBMax, vec3(modelMatrix * vec4(corner, 1.0f)));
  }

  transformedCentroid = transformedAABBMin + 0.5f * (transformedAABBMax - transformedAABBMin);
}

void Mesh::translate(vec3 t) {
  modelMatrix = glm::translate(modelMatrix, t);
  update();
}

void Mesh::rotateX(float angle) {
  modelMatrix = rotate(modelMatrix, radians(angle), vec3(1, 0, 0));
  update();
}

void Mesh::rotateY(float angle) {
  modelMatrix = rotate(modelMatrix, radians(angle), vec3(0, 1, 0));
  update();
}

void Mesh::rotateZ(float angle) {
  modelMatrix = rotate(modelMatrix, radians(angle), vec3(0, 0, 1));
  update();
}

void Mesh::scale(float scale) {
  modelMatrix = glm::scale(modelMatrix, vec3{scale});
  update();
}

void Mesh::center() {
  modelMatrix = glm::translate(modelMatrix, -geometry.centroid);
  update();
}