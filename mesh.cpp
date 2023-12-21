#include "mesh.hpp"

#include <iostream>

Mesh::Mesh(const Geometry& geometry, Material material) : geometry{geometry}, material{material} {
  update();
}

void Mesh::intersect(const Ray& ray, HitRecord& hitRecord, uint& count) const {
  Ray transformedRay = ray;
  transformedRay.origin = invModelMatrix * vec4(ray.origin, 1.0f);
  // Not normalized to handle scale transform
  transformedRay.direction = invModelMatrix * vec4(ray.direction, 0.0f);
  transformedRay.invDirection = 1.0f / transformedRay.direction;

  geometry.intersect(transformedRay, hitRecord, count);
}

void Mesh::update() {
  invModelMatrix = inverse(modelMatrix);
  normalMatrix = transpose(invModelMatrix);

  aabbMin = vec3{FLT_MAX};
  aabbMax = vec3{-FLT_MAX};

  for (auto& corner : geometry.corners) {
    aabbMin = min(aabbMin, vec3(modelMatrix * vec4(corner, 1.0f)));
    aabbMax = max(aabbMax, vec3(modelMatrix * vec4(corner, 1.0f)));
  }

  centroid = aabbMin + 0.5f * (aabbMax - aabbMin);
}

void Mesh::translate(vec3 t) {
  modelMatrix = glm::translate(modelMatrix, t);
  update();
}

void Mesh::rotateX(float angle) {
  modelMatrix = rotate(modelMatrix, angle, vec3(1, 0, 0));
  update();
}

void Mesh::rotateY(float angle) {
  modelMatrix = rotate(modelMatrix, angle, vec3(0, 1, 0));
  update();
}

void Mesh::rotateZ(float angle) {
  modelMatrix = rotate(modelMatrix, angle, vec3(0, 0, 1));
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

vec3 Mesh::getMin() const {
  return aabbMin;
}

vec3 Mesh::getMax() const {
  return aabbMax;
}